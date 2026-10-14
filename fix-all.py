#!/usr/bin/env python3
import struct

with open('models/euromillon-base-gpt2.gguf.G2M2.bak', 'rb') as f:
    data = bytearray(f.read())

print('File size:', len(data), 'bytes')

# Fix magic: G2M2 -> GGUF
data[0:4] = b'GGUF'

# Fix version: 33609216 -> 3
struct.pack_into('<I', data, 4, 3)

# Find all KV pairs and add padding + fix val_type
pos = 24
kv_pairs = []

while pos < len(data):
    if pos + 4 > len(data):
        break
    key_len = struct.unpack('<I', data[pos:pos+4])[0]
    key_start = pos + 4
    key_end = key_start + key_len
    if key_end + 8 > len(data):
        break
    val_type = struct.unpack('<I', data[key_end:key_end+4])[0]
    val_len = struct.unpack('<I', data[key_end+4:key_end+8])[0]
    kv_end = key_end + 8 + val_len
    if kv_end > len(data):
        break
    key = data[key_start:key_end].decode('utf-8', errors='replace')
    print('KV pair: key=' + key + ' val_type=' + str(val_type) + ' val_len=' + str(val_len))
    
    # Add 4 bytes of padding after key_len
    padding = b'\x00\x00\x00\x00'
    data[pos+4:pos+4] = padding
    
    # Fix val_type: 23040 -> 0 (string type)
    if val_type == 23040:
        data[key_end:key_end+4] = struct.pack('<I', 0)
        print('  Fixed val_type from 23040 to 0')
    
    kv_pairs.append((pos, key_end, kv_end))
    pos = kv_end + 4

print('Found', len(kv_pairs), 'KV pairs')

# Update header
data[16:24] = len(kv_pairs).to_bytes(8, 'little')

with open('models/euromillon-base-gpt2-CORRECTED.gguf', 'wb') as f:
    f.write(data)
print('Created CORRECTED file')