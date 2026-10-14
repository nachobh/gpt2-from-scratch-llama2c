#!/usr/bin/env python3
"""Debug del formato binario original."""
import os, sys, struct

binary_path = "models/euromillon-base-gpt2"
print(f"Archivo: {binary_path}")
print(f"Tamaño: {os.path.getsize(binary_path) / (1024*1024):.1f} MB")

with open(binary_path, 'rb') as f:
    header = f.read(5)
    print(f"Header: {header}")
    
    # Leer todos los valores de config (2 bytes cada uno)
    config_values = []
    for i in range(20):
        b1, b2 = f.read(2)
        val = struct.unpack('<H', bytes([b1, b2]))[0]
        config_values.append(val)
    
    print(f"Config values: {config_values}")
    
    # Leer cada tensor
    tensor_count = 0
    while True:
        try:
            name_len = struct.unpack('<H', f.read(2))[0]
            if name_len == 0 or name_len > 1000:
                break
            name = f.read(name_len).decode('utf-8').rstrip('\x00')
            
            # Leer dimensiones (4 valores de 2 bytes cada uno)
            dim0 = struct.unpack('<H', f.read(2))[0]
            dim1 = struct.unpack('<H', f.read(2))[0]
            dim2 = struct.unpack('<H', f.read(2))[0]
            dim3 = struct.unpack('<H', f.read(2))[0]
            
            # Calcular tamaño de datos
            if dim0 and dim1 and dim2 and dim3:
                data_len = dim0 * dim1 * dim2 * dim3 * 4
            elif dim0 and dim1:
                data_len = dim0 * dim1 * 4
            else:
                data_len = 4
            
            f.read(data_len)
            
            print(f"  Tensor #{tensor_count}: {name}")
            print(f"    Dims: ({dim0}, {dim1}, {dim2}, {dim3})")
            print(f"    Data size: {data_len / (1024*1024):.1f} MB")
            
            tensor_count += 1
            if tensor_count >= 10:
                break
        except Exception as e:
            print(f"  Fin o error: {e}")
            break