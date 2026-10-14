#!/usr/bin/env python3
import os, sys, struct

base_bin = "models/euromillon-base-gpt2"
print(f"Archivo: {base_bin}")
print(f"Tamaño total: {os.path.getsize(base_bin) / (1024*1024):.1f} MB")

with open(base_bin, 'rb') as f:
    header = f.read(5)
    print(f"Header (5 bytes): {header}")
    
    # Leer TODOS los valores de config (cada uno 2 bytes)
    values = []
    pos = 5
    while pos + 2 <= os.path.getsize(base_bin):
        b1, b2 = f.read(2)
        if not b1 or not b2:
            break
        val = struct.unpack('<H', bytes([b1, b2]))[0]
        values.append(val)
        pos += 2
        if val > 100:  # Probablemente nombre de tensor
            break
    
    print(f"Total valores leídos: {len(values)}")
    print(f"Primeros 20 valores: {values[:20]}")
    
    # Intentar leer tensores desde cada posición
    for test_pos in [15, 17, 20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 150, 200]:
        with open(base_bin, 'rb') as f2:
            f2.seek(test_pos)
            name_len_bytes = f2.read(2)
            if len(name_len_bytes) < 2:
                continue
            name_len = struct.unpack('<H', name_len_bytes)[0]
            if name_len > 0 and name_len < 500:
                name = f2.read(name_len).decode('utf-8', errors='replace').rstrip('\x00')
                if name and 'weight' in name:
                    print(f"  Posición {test_pos}: TENSOR ENCONTRADO - '{name}' (name_len={name_len})")