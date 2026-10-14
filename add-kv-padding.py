#!/usr/bin/env python3
import struct

with open('models/euromillon-base-gpt2.gguf', 'rb') as f:
    data = bytearray(f.read())

print('File size:', len(data), 'bytes')
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
    print('KV pair: key=' + key + ' key_len=' + str(key_len) + ' val_type=' + str(val_type) + ' val_len=' + str(val_len))
    kv_pairs.append((pos, key_end, kv_end))
    pos = kv_end

print('Found', len(kv_pairs), 'KV pairs')

for i in range(len(kv_pairs) - 1, -1, -1):
    kv_start, key_end, kv_end = kv_pairs[i]
    padding = b'\x00\x00\x00\x00'
    data[kv_start+4:kv_start+4] = padding
    for j in range(i + 1, len(kv_pairs)):
        s, e, end = kv_pairs[j]
        kv_pairs[j] = (s + 4, e + 4, end + 4)

print('Added padding to', len(kv_pairs), 'KV pairs')
print('New file size:', len(data), 'bytes')

with open('models/euromillon-base-gpt2-PADING.gguf', 'wb') as f:
    f.write(data)
print('Created ADDED-PADDING file')
