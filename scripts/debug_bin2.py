#!/usr/bin/env python3
import os, sys, struct

base_bin = "models/euromillon-base-gpt2"
print(f"Archivo: {base_bin}")
print(f"Tamaño total: {os.path.getsize(base_bin) / (1024*1024):.1f} MB")

with open(base_bin, 'rb') as f:
    data = f.read(200)  # Leer primeros 200 bytes
    print(f"\nPrimeros 200 bytes (hex):")
    for i in range(0, min(200, len(data)), 16):
        hex_str = ' '.join(f'{b:02x}' for b in data[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        print(f"  {i:04x}: {hex_str:<48} {ascii_str}")
    
    print(f"\nAnálisis estructural:")
    
    # Header (5 bytes): G2M2\x00
    header = data[0:5]
    print(f"  Header (5 bytes): {header} ({header.hex()})")
    
    # Después del header, leer valores de config (cada 2 bytes)
    pos = 5
    values = []
    for i in range(20):
        if pos + 2 > len(data):
            break
        val = struct.unpack('<H', data[pos:pos+2])[0]
        values.append(val)
        pos += 2
        if val == 0 and i > 2:
            break
    
    print(f"  Valores de config: {values}")
    print(f"  Posición después de config: {pos}")
    
    # A partir de aquí debería haber los tensores
    # Cada tensor: name_len (2 bytes) + name + dims (8 bytes) + data
    print(f"\n  Intentando leer tensores desde posición {pos}:")
    for tensor_idx in range(5):
        if pos + 2 > len(data):
            print(f"    Fin de datos en índice {tensor_idx}")
            break
        
        name_len = struct.unpack('<H', data[pos:pos+2])[0]
        pos += 2
        
        if name_len > 1000:
            print(f"    name_len grande ({name_len}), podría ser config")
            break
        
        if name_len > 0:
            name = data[pos:pos+name_len].decode('utf-8', errors='replace').rstrip('\x00')
            pos += name_len
        else:
            name = ""
        
        if pos + 8 <= len(data):
            d0, d1, d2, d3 = struct.unpack('<HHHH', data[pos:pos+8])
            pos += 8
            print(f"    Tensor {tensor_idx}: '{name}' dims=({d0},{d1},{d2},{d3})")
        else:
            print(f"    Tensor {tensor_idx}: '{name}' (sin dimensiones)")
            break