#!/usr/bin/env python3
"""Convertir modelo V2 (PT) a GGUF para llama-cli v1 (3a3edc9)"""

import torch
import struct
import json
from pathlib import Path
from collections import OrderedDict

# Configuración
INPUT_PT = 'models/euromillon-v2-gpt2.pt'
OUTPUT_GGUF = 'models/euromillon-v2-gpt2.gguf'

# Cargar modelo V2
print('Cargando modelo V2...')
weights = torch.load(INPUT_PT, weights_only=False)
model_state = weights['model_state']
config = weights['config']
dataset = weights['dataset']

print(f'Tensors: {len(model_state)}')

# Extraer configuración
vocab_size = dataset['vocab_size']
n_layers = config['n_layers']
n_heads = config['n_heads']
dim = config['dim']
hidden_dim = config['hidden_dim']
max_seq_len = config['max_seq_len']
norm_eps = config['norm_eps']

# Mapear nombres de tensors al formato esperado
TENSOR_MAP = {
    'tok_embeddings.weight': 0,
    'norm.weight': 1,
    'output.weight': 2,
}

for layer_idx in range(n_layers):
    TENSOR_MAP[f'layers.{layer_idx}.attention.wq.weight'] = 3 + layer_idx * 10
    TENSOR_MAP[f'layers.{layer_idx}.attention.wk.weight'] = 4 + layer_idx * 10
    TENSOR_MAP[f'layers.{layer_idx}.attention.wv.weight'] = 5 + layer_idx * 10
    TENSOR_MAP[f'layers.{layer_idx}.attention.wo.weight'] = 6 + layer_idx * 10
    TENSOR_MAP[f'layers.{layer_idx}.feed_forward.w1.weight'] = 7 + layer_idx * 10
    TENSOR_MAP[f'layers.{layer_idx}.feed_forward.w2.weight'] = 8 + layer_idx * 10
    TENSOR_MAP[f'layers.{layer_idx}.feed_forward.w3.weight'] = 9 + layer_idx * 10
    TENSOR_MAP[f'layers.{layer_idx}.attention_norm.weight'] = 1 + layer_idx * 10
    TENSOR_MAP[f'layers.{layer_idx}.ffn_norm.weight'] = 2 + layer_idx * 10

n_tensors = len(model_state)

# Preparar lista ordenada de pesos
ordered_weights = []
for key in sorted(TENSOR_MAP.keys(), key=lambda k: TENSOR_MAP[k]):
    if key in model_state:
        ordered_weights.append(model_state[key].float().contiguous())
    else:
        print(f'  WARNING: Tensor {key} NOT FOUND')

print(f'Pesos ordenados: {len(ordered_weights)}')

# Crear lista de tokens
tokens = sorted(set(chr(i) for i in range(256)))
tokens = [t for t in tokens if t.isprintable()][:vocab_size]
while len(tokens) < vocab_size:
    tokens.append(chr(255))

tokens_list = ' '.join(f'chr({ord(t)})' if ord(t) > 127 else f'"{t}"' for t in tokens[:vocab_size])
scores_list = ' '.join(['0.0'] * vocab_size)
types_list = ' '.join(['1'] * vocab_size)

# === ESCRIBIR GGUF (FORMATO PARA LLAMA-CLI v1) ===
print(f'\nEscribiendo GGUF...')

# Header - usar formato compatible con llama-cli v1
header_size = 24  # magic(4) + version(4) + tensor_count(8) + kv_count(8)

# KV pairs - usar key_len de 4 bytes (I/uint32) como espera llama-cli v1
kv_pairs = [
    ('general.architecture', 'gpt2'),
    ('gpt2.block_count', str(n_layers)),
    ('gpt2.context_length', str(max_seq_len)),
    ('gpt2.embedding_length', str(dim)),
    ('gpt2.feed_forward_length', str(hidden_dim)),
    ('gpt2.attention.head_count', str(n_heads)),
    ('gpt2.attention.layer_norm_epsilon', str(config['norm_eps'])),
    ('gpt2.attention.key_length', str(dim // n_heads)),
    ('gpt2.attention.value_length', str(dim // n_heads)),
    ('tokenizer.ggml.tokens', tokens_list),
    ('tokenizer.ggml.scores', scores_list),
    ('tokenizer.ggml.token_type', types_list),
]

# Calcular sizes - usar key_len de 4 bytes para KV pairs
kv_size = 0
for key, value in kv_pairs:
    key_bytes = key.encode('utf-8')
    value_bytes = value.encode('utf-8')
    kv_size += 4 + len(key_bytes) + 4 + 4 + len(value_bytes)  # key_len(4) + key + val_type(4) + val_len(4) + value

# Cada header de tensor es 24 bytes
tensor_header_size = 24 * n_tensors

# Calcular offsets de datos
data_start_pos = header_size + kv_size + tensor_header_size
tensor_offsets = []
for i, weight in enumerate(ordered_weights):
    tensor_offsets.append(data_start_pos)
    data_start_pos += weight.numel() * 4

# Calcular cada tensor's data size
tensor_data_sizes = []
for i in range(n_tensors):
    n_elements = len(ordered_weights[i])
    data_size = n_elements * 4  # Float32 = 4 bytes
    tensor_data_sizes.append(data_size)

# Escribir archivo
output_path = 'models/euromillon-v2-gpt2.gguf'
print(f'Writing to: {output_path}')
print(f'Header size: {header_size} bytes')
print(f'KV pairs size: {kv_size} bytes')
print(f'Tensor headers size: {tensor_header_size} bytes')
print(f'Tensor data starts at: {data_start_pos} bytes')

with open(output_path, 'wb') as f:
    # 1. HEADER
    f.write(b'GGUF')
    f.write(struct.pack('<I', 3))  # version
    f.write(struct.pack('<Q', n_tensors))
    f.write(struct.pack('<Q', len(kv_pairs)))
    
    # 2. KV PAIRS (key_len = 4 bytes para llama-cli v1)
    for key, value in kv_pairs:
        key_bytes = key.encode('utf-8')
        value_bytes = value.encode('utf-8')
        f.write(struct.pack('<I', len(key_bytes)))  # key_len as uint32 (4 bytes) - llama-cli v1 expects this
        f.write(key_bytes)
        f.write(struct.pack('<I', 0))  # String value type
        f.write(struct.pack('<I', len(value_bytes)))  # Value length (uint32, 4 bytes)
        f.write(value_bytes)
    
    # 3. TENSOR HEADERS (24 bytes each)
    for i, weight in enumerate(ordered_weights):
        n_elements = weight.numel()
        
        # Determine shape
        if i == 0:  # tok_embeddings
            shape = (vocab_size, dim)
        elif i == 1:  # norm
            shape = (dim,)
        elif i == 2:  # output
            shape = (vocab_size, dim)
        elif i in [3, 4, 5, 6, 13, 14, 15, 16, 23, 24, 25, 26, 33, 34, 35, 36, 43, 44, 45, 46, 53, 54, 55, 56, 63, 64, 65, 66]:
            shape = (dim, dim)  # wq, wk, wv, wo
        elif i in [7, 8, 9, 17, 18, 19, 27, 28, 29, 37, 38, 39, 47, 48, 49, 57, 58, 59, 67, 68, 69]:
            shape = (hidden_dim, dim)  # w1, w3
        elif i in [10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62, 70, 71, 72, 73]:
            shape = (dim,)  # attention_norm, ffn_norm
        else:
            shape = (n_elements,)
        
        tensor_name = f'tensor.{i}'
        name_bytes = tensor_name.encode('utf-8')
        ndim = len(shape)
        
        # Write tensor header (24 bytes each)
        f.write(struct.pack('<I', 4))  # name_len = 4 (always "tensor")
        f.write(b'tensor')  # name = "tensor" (4 bytes)
        f.write(struct.pack('<I', ndim))
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
    
    # 4. TENSOR DATA
    for weight in ordered_weights:
        f.write(weight.cpu().numpy().tobytes())

file_size = Path(output_path).stat().st_size
print(f'\nFile size: {file_size / (1024*1024):.1f} MB')
print('Conversion complete!')