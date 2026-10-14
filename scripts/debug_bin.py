#!/usr/bin/env python3
import os, sys, struct

base_bin = "models/euromillon-base-gpt2"
print(f"Archivo: {base_bin}")
print(f"Tamaño: {os.path.getsize(base_bin) / (1024*1024):.1f} MB")

with open(base_bin, 'rb') as f:
    # Header (5 bytes)
    header = f.read(5)
    print(f"Header (5 bytes): {header}")
    
    # Leer 100 valores de config (cada 2 bytes)
    values = []
    for i in range(100):
        b1, b2 = f.read(2)
        if not b1 or not b2:
            print(f"  Fin de archivo en posición {f.tell()}")
            break
        val = struct.unpack('<H', bytes([b1, b2]))[0]
        values.append(val)
        if val == 0 and i > 2:
            break
    
    print(f"Valores leídos: {values[:15]}")
    print(f"Posición después de config: {f.tell()}")
    
    # Leer el primer tensor
    print("\nLeyendo primer tensor...")
    name_len_bytes = f.read(2)
    if len(name_len_bytes) < 2:
        print("  No se pudo leer name_len")
    else:
        name_len = struct.unpack('<H', name_len_bytes)[0]
        print(f"  name_len: {name_len}")
        
        if name_len > 0 and name_len < 500:
            name = f.read(name_len).decode('utf-8').rstrip('\x00')
            print(f"  Nombre: '{name[:50]}...'")
        
        # Leer dimensiones
        d0 = struct.unpack('<H', f.read(2))[0]
        d1 = struct.unpack('<H', f.read(2))[0]
        d2 = struct.unpack('<H', f.read(2))[0]
        d3 = struct.unpack('<H', f.read(2))[0]
        print(f"  Dimensiones: ({d0}, {d1}, {d2}, {d3})")
        
        # Calcular tamaño de datos
        if d0 and d1 and d2 and d3:
            data_len = d0 * d1 * d2 * d3 * 4
        elif d0 and d1:
            data_len = d0 * d1 * 4
        else:
            data_len = 4
        print(f"  Tamaño de datos: {data_len / (1024*1024):.1f} MB")
        
        # Leer datos
        data = f.read(data_len)
        print(f"  Datos leídos: {len(data)} bytes")