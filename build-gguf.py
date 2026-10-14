#!/usr/bin/env python3
"""Convert PyTorch .pt model to GGUF format (v2 compatible).

GGUF v2 layout:
1. Header (magic + version + tensor_count + kv_count)
2. KV pairs
3. TENSOR HEADERS (name + shape + type + offset)
4. TENSOR DATA (raw binary)
"""

import zipfile
import struct
import numpy as np
from pathlib import Path

# GGUF constants
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

# Model configuration
MODEL_CONFIG = {
    'dim': 512,
    'n_layers': 8,
    'n_heads': 8,
    'n_kv_heads': 8,
    'hidden_dim': 2048,
    'max_seq_len': 128,
    'vocab_size': 45,
    'norm_eps': '1e-5',
}

# Extract weights
print('Extracting weights...')
weights = {}
with zipfile.ZipFile('models/euromillon-base-gpt2.pt', 'r') as z:
    for i in range(74):
        data_path = f'euromillon-base-gpt2/data/{i}'
        if data_path in z.namelist():
            with z.open(data_path) as f:
                data = f.read()
            arr = np.frombuffer(data, dtype=np.float32)
            weights[i] = arr
print(f'Found {len(weights)} tensors')

# Count tensors and KV pairs
n_tensors = len(weights)
tokens_list = [f'chr({i})' for i in range(MODEL_CONFIG['vocab_size'])]
scores_list = ['0.0'] * MODEL_CONFIG['vocab_size']
types_list = ['1'] * MODEL_CONFIG['vocab_size']

kv_pairs = [
    ('general.architecture', 'gpt2'),
    ('gpt2.block_count', str(MODEL_CONFIG['n_layers'])),
    ('gpt2.context_length', str(MODEL_CONFIG['max_seq_len'])),
    ('gpt2.embedding_length', str(MODEL_CONFIG['dim'])),
    ('gpt2.feed_forward_length', str(MODEL_CONFIG['hidden_dim'])),
    ('gpt2.attention.head_count', str(MODEL_CONFIG['n_heads'])),
    ('gpt2.attention.layer_norm_epsilon', MODEL_CONFIG['norm_eps']),
    ('gpt2.attention.key_length', str(MODEL_CONFIG['dim'] // MODEL_CONFIG['n_heads'])),
    ('gpt2.attention.value_length', str(MODEL_CONFIG['dim'] // MODEL_CONFIG['n_heads'])),
    ('tokenizer.ggml.tokens', ' '.join(tokens_list)),
    ('tokenizer.ggml.scores', ' '.join(scores_list)),
    ('tokenizer.ggml.token_type', ' '.join(types_list)),
]
n_kv_pairs = len(kv_pairs)

# Calculate each tensor's data size
dim = MODEL_CONFIG['dim']
hidden = MODEL_CONFIG['hidden_dim']
vocab = MODEL_CONFIG['vocab_size']

# Calculate sizes for each tensor to determine offsets
header_size = 24  # magic(4) + version(4) + tensor_count(8) + kv_count(8)

# Calculate KV pairs size
kv_size = 0
for key, value in kv_pairs:
    key_bytes = key.encode('utf-8')
    value_bytes = value.encode('utf-8')
    kv_size += 8 + len(key_bytes) + 4 + 4 + len(value_bytes)  # key_len(8) + key + val_type(4) + val_len(4) + value

# Each header is 24 bytes: name_len(4) + name(4 + ndim(4) + shape(8) + type(4) + offset(4)
tensor_header_size = 24 * n_tensors

# Total position where tensor data starts
data_start_pos = header_size + kv_size + tensor_header_size

# Calculate each tensor's data size
tensor_data_sizes = []
for i in range(n_tensors):
    n_elements = len(weights[i])
    # Each element is 4 bytes (float32)
    tensor_data_sizes.append(n_elements * 4)

# Calculate offsets for each tensor
tensor_offsets = []
current_offset = data_start_pos
for size in tensor_data_sizes:
    tensor_offsets.append(current_offset)
    current_offset += size

# Write the file
output_path = 'models/euromillon-base-gpt2-NEW.gguf'
print(f'Writing GGUF file...')
print(f'  Header size: {header_size} bytes')
print(f'  KV pairs size: {kv_size} bytes')
print(f'  Tensor headers size: {tensor_header_size} bytes')
print(f'  Tensor data starts at: {data_start_pos} bytes')

with open(output_path, 'wb') as f:
    # 1. HEADER
    header = b''
    header += GGUF_MAGIC
    header += struct.pack('<I', GGUF_VERSION)
    header += struct.pack('<Q', n_tensors)
    header += struct.pack('<Q', n_kv_pairs)
    f.write(header)
    
    # KV pairs
    for key, value in kv_pairs:
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<Q', len(key_bytes)))  # Use Q (uint64, 8 bytes) - llama-cli reads 8 bytes
        f.write(key_bytes)
        f.write(struct.pack('<I', 0))  # String value type
        value_bytes = value.encode('utf-8')
        f.write(struct.pack('<I', len(value_bytes)))  # Value length (uint32, 4 bytes) - llama-cli v1 expects this
        f.write(value_bytes)
    
    # 3. TENSOR HEADERS (metadata with offsets)
    print(f'  Writing tensor headers...')
    for i in range(n_tensors):
        tensor = weights[i]
        n_elements = len(tensor)
        
        # Determine shape
        if i == 0:
            shape = (vocab, dim)
        elif i == 1:
            shape = (dim,)
        elif i in [2, 3, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 50, 51, 58, 59, 66, 67, 72]:
            shape = (dim,)
        elif i in [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63, 68, 69, 70, 71]:
            shape = (dim, dim)
        elif i in [8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 64, 65, 73]:
            shape = (hidden, dim)
        else:
            shape = (n_elements,)
        
        tensor_name = f'tensor.{i}'
        name_bytes = tensor_name.encode('utf-8')
        ndim = len(shape)
        
        # Write tensor header (24 bytes each)
        # Format: name_len(4) + name(4) + ndim(4) + shape(8) + type(4) + offset(4) = 28 bytes
        f.write(struct.pack('<I', 4))  # name_len = 4 (always "tensor")
        f.write(b'tensor')  # name = "tensor" (4 bytes)
        f.write(struct.pack('<I', ndim))  # ndim
        if len(shape) >= 1:
            f.write(struct.pack('<Q', shape[0]))
        else:
            f.write(struct.pack('<Q', 1))
        if len(shape) >= 2:
            f.write(struct.pack('<Q', shape[1]))
        else:
            f.write(struct.pack('<Q', 1))
        f.write(struct.pack('<I', 1))  # Float32
        f.write(struct.pack('<I', tensor_offsets[i] >> 2))  # Offset in 4-byte units
    
    # 4. TENSOR DATA (raw binary)
    print(f'  Writing tensor data...')
    for i in range(n_tensors):
        tensor = weights[i]
        f.write(tensor.tobytes())

file_size = Path(output_path).stat().st_size
print(f'\nFile size: {file_size / (1024*1024):.1f} MB')
print('Conversion complete!')
