#!/usr/bin/env python3
"""Parsea el archivo binario original para extraer la config."""
import os, sys, struct

binary_path = "models/euromillon-base-gpt2"
print(f"Archivo: {binary_path}")
print(f"Tamaño: {os.path.getsize(binary_path) / (1024*1024):.1f} MB")

with open(binary_path, 'rb') as f:
    header = f.read(5)
    print(f"Header: {header}")
    
    # Leer config como valores packed (little-endian, cada uno 4 bytes)
    n_layer = struct.unpack('<I', f.read(4))[0]
    n_head = struct.unpack('<I', f.read(4))[0]
    n_kv_heads = struct.unpack('<I', f.read(4))[0]
    vocab_size = struct.unpack('<I', f.read(4))[0]
    max_seq_len = struct.unpack('<I', f.read(4))[0]
    
    print(f"Config (packed):")
    print(f"  n_layer={n_layer}, n_head={n_head}, n_kv_heads={n_kv_heads}")
    print(f"  vocab_size={vocab_size}, max_seq_len={max_seq_len}")
    
    # Leer cada tensor
    while True:
        try:
            name_len = struct.unpack('<I', f.read(4))[0]
            if name_len == 0 or name_len > 1000:
                break
            name = f.read(name_len).decode('utf-8').rstrip('\x00')
            
            # Leer tamaño del tensor (4 valores little-endian)
            dim0 = struct.unpack('<I', f.read(4))[0]
            dim1 = struct.unpack('<I', f.read(4))[0]
            dim2 = struct.unpack('<I', f.read(4))[0]
            dim3 = struct.unpack('<I', f.read(4))[0]
            
            # Calcular tamaño de datos
            if dim0 and dim1 and dim2 and dim3:
                data_len = dim0 * dim1 * dim2 * dim3 * 4  # float32 = 4 bytes
            elif dim0 and dim1:
                data_len = dim0 * dim1 * 4
            else:
                data_len = 4
            
            f.read(data_len)
            
            print(f"  Tensor: {name} dims=({dim0},{dim1},{dim2},{dim3}) size={data_len/1024/1024:.1f}MB")
        except Exception as e:
            print(f"  Fin o error: {e}")
            break