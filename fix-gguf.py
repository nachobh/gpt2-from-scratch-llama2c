#!/usr/bin/env python3
"""Fix GGUF files to match llama-cli v1 expectations"""
import struct

def fix_gguf(input_path, output_path, n_tensors):
    with open(input_path, 'rb') as f:
        data = bytearray(f.read())
    
    # Fix magic
    data[0:4] = b'GGUF'
    
    # Fix version
    struct.pack_into('<I', data, 4, 3)
    
    # Fix tensor count
    struct.pack_into('<Q', data, 8, n_tensors)
    
    with open(output_path, 'wb') as f:
        f.write(data)
    print(f'Fixed: {input_path} -> {output_path}')

# Fix V1 (original G2M2.bak - has only 1 KV pair and wrong format)
# The original has: magic=G2M2, version=33609216, tensors=34359738880, kv=193273528328
fix_gguf('models/euromillon-base-gpt2.gguf.G2M2.bak', 'models/euromillon-base-gpt2-V1.gguf', 74)

# V2 already has correct format from build-gguf-v2.py
# Just verify and fix if needed
with open('models/euromillon-v2-gpt2.gguf', 'rb') as f:
    data = f.read(24)
magic = data[:4]
version = struct.unpack('<I', data[4:8])[0]
n_tensors = struct.unpack('<Q', data[8:16])[0]
n_kv = struct.unpack('<Q', data[16:24])[0]
print(f'V2: magic={magic}, version={version}, tensors={n_tensors}, kv={n_kv}')
