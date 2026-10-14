#!/usr/bin/env python3
"""Fix V1 GGUF header to match V2 format that works with llama-server"""

import struct
import sys

# Read ORIGINAL file
with open('models/euromillon-base-gpt2.gguf.G2M2.bak', 'rb') as f:
    data = bytearray(f.read())

print(f'Original file size: {len(data)} bytes')

# Find the KV pair end
kv_start = 24
key_len = struct.unpack('<I', data[kv_start:kv_start+4])[0]
key = data[kv_start+4:kv_start+4+key_len].decode('utf-8')
val_start = kv_start + 4 + key_len
val_type = struct.unpack('<I', data[val_start:val_start+4])[0]
val_len = struct.unpack('<I', data[val_start+4:val_start+8])[0]
kv_end = val_start + 8 + val_len

print(f'KV pair found: key={key}, val_type=0x{val_type:04x}, val_len={val_len}')
print(f'KV ends at: {kv_end}')

# Fix magic and version
data[0:4] = b'GGUF'
struct.pack_into('<I', data, 4, 3)

# Add 4 bytes of padding after key_len
padding = b'\x00\x00\x00\x00'
data[kv_start+4:kv_start+4] = padding

# Fix val_type and val_len
val_start_fixed = kv_start + 4 + key_len + 4
data[val_start_fixed:val_start_fixed+4] = struct.pack('<I', 0)
data[val_start_fixed+4:val_start_fixed+8] = struct.pack('<I', 2)

# Update header kv count
data[16:24] = (1).to_bytes(8, 'little')

with open('models/euromillon-base-gpt2-FIXED.gguf', 'wb') as f:
    f.write(data)

print(f'Created FIXED file: {len(data)} bytes')
print('Done!')
