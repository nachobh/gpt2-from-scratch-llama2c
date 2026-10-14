#!/usr/bin/env python3
"""Convierte el modelo SWE-Zero al formato binario de llama2.c."""
import os, sys, struct, json
import torch
import safetensors.torch as st

LLAMA2C_DIR = os.path.join(os.path.dirname(__file__), '..', 'llama2.c')
sys.path.insert(0, LLAMA2C_DIR)

def main():
    safetensors_path = "models/euromillon-swe-zero.safetensors"
    binary_path = "models/euromillon-swe-zero.bin"
    base_binary = "models/euromillon-base-gpt2"
    
    print(f"Convirtiendo {safetensors_path} -> {binary_path}")
    
    # Cargar modelo SWE-Zero
    print("Cargando modelo SWE-Zero...")
    state_dict = st.load_file(safetensors_path)
    
    # Leer la estructura del binario original
    print("Leyendo estructura del binario original...")
    with open(base_binary, 'rb') as f:
        header = f.read(5)
        print(f"  Header: {header}")
        
        # Leer todos los valores de config (cada uno 2 bytes)
        all_values = []
        # El primer valor (124) es la longitud del nombre del primer tensor
        first_name_len = struct.unpack('<H', f.read(2))[0]
        print(f"  Primer nombre de tensor length: {first_name_len}")
    
        # El resto son valores de config
        for i in range(19):  # Leer 19 valores más después del primero
            b1, b2 = f.read(2)
            val = struct.unpack('<H', bytes([b1, b2]))[0]
            all_values.append(val)
            if val == 0 and i > 2:
                break
    
        # Añadir el primer valor al inicio
        all_values.insert(0, first_name_len)
    
        print(f"  Valores leídos: {all_values[:15]}")
    
        # Leer el nombre del primer tensor del archivo original
        first_name = f.read(first_name_len).decode('latin-1').rstrip('\x00')
        print(f"  Primer tensor: '{first_name}'")
        
        # Leer dimensiones del primer tensor
        dim0 = struct.unpack('<H', f.read(2))[0]
        dim1 = struct.unpack('<H', f.read(2))[0]
        dim2 = struct.unpack('<H', f.read(2))[0]
        dim3 = struct.unpack('<H', f.read(2))[0]
        print(f"  Primer tensor dims: ({dim0}, {dim1}, {dim2}, {dim3})")
        
        # Calcular tamaño de datos
        if dim0 and dim1 and dim2 and dim3:
            data_len = dim0 * dim1 * dim2 * dim3 * 4
        elif dim0 and dim1:
            data_len = dim0 * dim1 * 4
        else:
            data_len = 4
        print(f"  Primer tensor data size: {data_len / (1024*1024):.1f} MB")
        
        # Saltar los datos del primer tensor
        f.seek(data_len, 1)
        
        # Leer siguiente tensor
        next_name_len = struct.unpack('<H', f.read(2))[0]
        
        # Leer todos los tensores del archivo original
        print("Leyendo tensores del archivo original...")
        tensors_original = []
        while True:
            try:
                name_len_bytes = f.read(2)
                if len(name_len_bytes) < 2:
                    break
                name_len = struct.unpack('<H', name_len_bytes)[0]
                
                if name_len > 2000:
                    break
                
                if name_len > 0 and name_len < 500:
                    name = f.read(name_len).decode('utf-8').rstrip('\x00')
                else:
                    name = f.read(name_len).decode('latin-1').rstrip('\x00')
                
                d0 = struct.unpack('<H', f.read(2))[0]
                d1 = struct.unpack('<H', f.read(2))[0]
                d2 = struct.unpack('<H', f.read(2))[0]
                d3 = struct.unpack('<H', f.read(2))[0]
                
                if d0 and d1 and d2 and d3:
                    data_len = d0 * d1 * d2 * d3 * 4
                elif d0 and d1:
                    data_len = d0 * d1 * 4
                else:
                    data_len = 4
                
                tensor_data = f.read(data_len)
                
                if name:
                    print(f"  Tensor: {name} dims=({d0},{d1},{d2},{d3})")
                    tensors_original.append({
                        'name': name,
                        'dims': (d0, d1, d2, d3),
                        'data': tensor_data,
                        'size': data_len
                    })
            except Exception as e:
                print(f"  Fin o error: {e}")
                break
        
        total_tensors = len(tensors_original)
        print(f"  Total de tensores leídos: {total_tensors}")
    
    # Generar nuevo binario con la misma estructura
    print(f"Generando formato binario: {binary_path}")
    with open(binary_path, 'wb') as f:
        # Header
        f.write(header)
        
        # Config
        for val in all_values[:20]:
            f.write(struct.pack('<H', val))
        
        # Escribir cada tensor
        for tensor_info in tensors_original:
            name = tensor_info['name']
            dims = tensor_info['dims']
            
            # Nombre del tensor
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<H', len(name_bytes)))
            f.write(name_bytes)
            
            # Dimensiones
            for d in dims:
                f.write(struct.pack('<H', d))
            
            # Datos del tensor (usar los datos del modelo SWE-Zero si están disponibles)
            if name in state_dict:
                tensor_float32 = state_dict[name].float()
                f.write(tensor_float32.cpu().numpy().tobytes('C'))
            else:
                # Si no está en el modelo SWE-Zero, usar los datos originales
                f.write(tensor_info['data'])
    
    print(f"  Binario: {os.path.getsize(binary_path) / (1024*1024):.1f} MB")
    print("  Conversión completada")

if __name__ == "__main__":
    main()