#!/usr/bin/env python3
"""Debug detallado del formato binario original."""
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
        val = struct.unpack('<H', bytes([b1,b2]))[0]
        config_values.append(val)
    
    print(f"Config values: {config_values}")
    
    # El primer valor parece ser la longitud del nombre del primer tensor
    first_name_len = config_values[0]
    print(f"Primer nombre de tensor length: {first_name_len}")
    
    # Leer el nombre del primer tensor
    first_name = f.read(first_name_len).decode('utf-8').rstrip('\x00')
    print(f"Primer tensor: '{first_name}'")
    
    # Leer dimensiones del primer tensor
    dim0 = struct.unpack('<H', f.read(2))[0]
    dim1 = struct.unpack('<H', f.read(2))[0]
    dim2 = struct.unpack('<H', f.read(2))[0]
    dim3 = struct.unpack('<H', f.read(2))[0]
    print(f"Primer tensor dims: ({dim0}, {dim1}, {dim2}, {dim3})")
    
    # Calcular tamaño de datos
    if dim0 and dim1 and dim2 and dim3:
        data_len = dim0 * dim1 * dim2 * dim3 * 4
    elif dim0 and dim1:
        data_len = dim0 * dim1 * 4
    else:
        data_len = 4
    print(f"Primer tensor data size: {data_len / (1024*1024):.1f} MB")
    
    # Saltar los datos del primer tensor
    f.seek(data_len, 1)
    
    # Leer siguiente tensor
    next_name_len = struct.unpack('<H', f.read(2))[0]
    print(f"Segundo nombre de tensor length: {next_name_len}")
    
    if next_name_len > 0 and next_name_len < 1000:
        next_name = f.read(next_name_len).decode('utf-8').rstrip('\x00')
        print(f"Segundo tensor: '{next_name}'")