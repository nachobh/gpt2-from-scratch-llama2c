#!/usr/bin/env python3
"""
Convert PyTorch .pt model to GGUF format.

This script reads the PyTorch checkpoint, extracts the model weights,
and writes them in GGUF format.

Usage:
    python3 convert-pt-to-gguf.py <input.pt> <output.gguf>
"""

import zipfile
import struct
import json
import sys
import math
from pathlib import Path

# GGUF constants
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3
GGUF_BYTE_ORDER = 0x00000001  # Little-endian

# Model configuration (from meta.json)
MODEL_CONFIG = {
    "dim": 512,
    "n_layers": 8,
    "n_heads": 8,
    "n_kv_heads": 8,
    "hidden_dim": 2048,  # dim * 4
    "max_seq_len": 128,
    "vocab_size": 45,
    "norm_eps": 1e-5,
}

# Weight naming convention for GPT-2
# Based on the weights found in the .pt file (74 tensors total)
# Let's map them to GGUF naming conventions


def extract_weights_from_pt(pt_path):
    """Extract weights from PyTorch ZIP format."""
    weights = {}
    
    with zipfile.ZipFile(pt_path, 'r') as z:
        # Read metadata
        with z.open('euromillon-base-gpt2/.format_version') as f:
            version = int(f.read().decode())
        
        with z.open('euromillon-base-gpt2/byteorder') as f:
            byteorder = f.read().decode().strip()
        
        with z.open('euromillon-base-gpt2/version') as f:
            pt_version = f.read().decode().strip()
        
        print(f"PyTorch format version: {version}")
        print(f"Byte order: {byteorder}")
        print(f"PyTorch version: {pt_version}")
        
        # Extract each weight
        for i in range(74):
            data_path = f'euromillon-base-gpt2/data/{i}'
            if data_path in z.namelist():
                with z.open(data_path) as f:
                    data = f.read()
                
                # Calculate tensor shape based on size
                # Each float32 is 4 bytes
                n_elements = len(data) // 4
                print(f"  Weight {i}: {n_elements} floats ({len(data)} bytes)")
                
                # Decode as float32
                import numpy as np
                arr = np.frombuffer(data, dtype=np.float32)
                weights[i] = arr
    
    return weights


def create_gguf_header(n_tensors, n_kv_pairs):
    """Create GGUF header bytes."""
    header = b''
    # Magic
    header += GGUF_MAGIC
    # Version
    header += struct.pack('<I', GGUF_VERSION)
    # Tensor count
    header += struct.pack('<Q', n_tensors)
    # KV count
    header += struct.pack('<Q', n_kv_pairs)
    return header


def get_tensor_shape(tensor, idx):
    """Determine tensor shape from size and index."""
    # Common shapes for GPT-2:
    # - Embedding: (vocab_size, dim) = (45, 512) = 23,040
    # - Linear layers: (out_features, in_features)
    # - LayerNorm: (dim,) = 512
    
    n_elements = len(tensor)
    dim = MODEL_CONFIG["dim"]
    hidden = MODEL_CONFIG["hidden_dim"]
    vocab = MODEL_CONFIG["vocab_size"]
    
    # Layer 0: token embedding
    if idx == 0:
        return (vocab, dim)
    
    # Layer 1: rms_final_weight (last layer)
    if idx == 1:
        return (dim,)
    
    # LayerNorm weights (each layer has 2)
    if idx in [2, 3, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 50, 51, 58, 59, 66, 67, 72, 73]:
        return (dim,)
    
    # Attention weights (each layer has 4)
    if idx in [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63, 68, 69, 70, 71]:
        return (dim, dim)
    
    # FFN weights (each layer has 2)
    if idx in [8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 64, 65, 72, 73]:
        return (hidden, dim)
    
    # Output layer (last)
    if idx == 73:
        return (vocab, dim)
    
    # Default: try to infer from size
    if n_elements == vocab * dim:
        return (vocab, dim)
    elif n_elements == dim * dim:
        return (dim, dim)
    elif n_elements == hidden * dim:
        return (hidden, dim)
    elif n_elements == dim:
        return (dim,)
    else:
        # Try to find factors
        for d1 in range(1, min(n_elements, 1000)):
            if n_elements % d1 == 0:
                d2 = n_elements // d1
                if d1 <= dim * 4 and d2 <= dim * 4:
                    return (d1, d2)
    
    return (n_elements,)


def write_gguf(weights, output_path):
    """Write weights in GGUF format."""
    print(f"\nWriting GGUF file: {output_path}")
    
    # Count tensors and KV pairs
    n_tensors = len(weights)
    
    # KV pairs for metadata
    kv_pairs = [
        ("general.architecture", "gpt2"),
        ("gpt2.block_count", MODEL_CONFIG["n_layers"]),
        ("gpt2.context_length", MODEL_CONFIG["max_seq_len"]),
        ("gpt2.embedding_length", MODEL_CONFIG["dim"]),
        ("gpt2.feed_forward_length", MODEL_CONFIG["hidden_dim"]),
        ("gpt2.attention.head_count", MODEL_CONFIG["n_heads"]),
        ("gpt2.attention.layer_norm_epsilon", str(MODEL_CONFIG["norm_eps"])),
        ("gpt2.attention.key_length", MODEL_CONFIG["dim"] // MODEL_CONFIG["n_heads"]),
        ("gpt2.attention.value_length", MODEL_CONFIG["dim"] // MODEL_CONFIG["n_heads"]),
        ("tokenizer.ggml.tokens", " ".join([f"chr({i})" for i in range(MODEL_CONFIG["vocab_size"])])),
        ("tokenizer.ggml.scores", " ".join(["0.0"] * MODEL_CONFIG["vocab_size"])),
        ("tokenizer.ggml.token_type", " ".join(["1"] * MODEL_CONFIG["vocab_size"])),
    ]
    
    n_kv_pairs = len(kv_pairs)
    
    # Write header
    with open(output_path, 'wb') as f:
        # Header
        header = create_gguf_header(n_tensors, n_kv_pairs)
        f.write(header)
        
        # KV pairs
        for key, value in kv_pairs:
            key_bytes = key.encode('utf-8')
            # Key length + key
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)
            # Value type (0 = string) + value
            f.write(struct.pack('<I', 0))
            value_bytes = value.encode('utf-8')
            f.write(struct.pack('<Q', len(value_bytes)))
            f.write(value_bytes)
        
        # Tensors
        for i in range(n_tensors):
            tensor = weights[i]
            shape = get_tensor_shape(tensor, i)
            
            # Tensor name (simplified)
            tensor_name = f"token.{i}"
            
            # Write tensor metadata
            # Name length + name
            name_bytes = tensor_name.encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            
            # Number of dimensions
            ndim = len(shape)
            f.write(struct.pack('<I', ndim))
            
            # Shape
            for dim in shape:
                f.write(struct.pack('<Q', dim))
            
            # Data type (1 = float32)
            f.write(struct.pack('<I', 1))
            
            # Data
            f.write(tensor.tobytes())
    
    # Calculate file size
    file_size = Path(output_path).stat().st_size
    print(f"File size: {file_size / (1024*1024):.1f} MB")


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 convert-pt-to-gguf.py <input.pt> <output.gguf>")
        sys.exit(1)
    
    pt_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Reading weights from: {pt_path}")
    weights = extract_weights_from_pt(pt_path)
    print(f"Found {len(weights)} tensors")
    
    write_gguf(weights, output_path)
    print("\nConversion complete!")


if __name__ == "__main__":
    main()
