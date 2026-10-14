#!/usr/bin/env python3
import struct

with open('models/euromillon-base-gpt2.gguf', 'rb') as f:
    data = bytearray(f.read())

print(f'File size: {len(data)} bytes')

pos = 24
kv_count = 0
total_padding = 0

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
    kv_end = key_end + 4 + 4 + val_len
    if kv_end > len(data):
        break
    padding = b'\x00\x00\x00\x00'
    data[pos+4:pos+4] = padding
    total_padding += 4
    pos = kv_end + 4
    kv_count += 1

print(f'Added padding to {kv_count} KV pairs')
print(f'Total padding: {total_padding} bytes')
print(f'New size: {len(data)} bytes')

data[16:24] = kv_count.to_bytes(8, 'little')
with open('models/euromillon-base-gpt2-ADDED-PADDING.gguf', 'wb') as f:
    f.write(data)
print('Created ADDED-PADDING file')
