#!/usr/bin/env python3
"""Convierte el modelo SWE-Zero desde .pt al formato binario de llama2.c."""
import os, sys, struct, json, torch

LLAMA2C_DIR = os.path.join(os.path.dirname(__file__), '..', 'llama2.c')
sys.path.insert(0, LLAMA2C_DIR)

def main():
    pt_path = "models/euromillon-swe-zero.pt"
    bin_path = "models/euromillon-swe-zero.bin"
    base_bin = "models/euromillon-base-gpt2"
    
    print(f"Convirtiendo {pt_path} -> {bin_path}")
    
    # Cargar modelo SWE-Zero desde .pt
    print("Cargando modelo SWE-Zero...")
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state', checkpoint.get('model', {}))
    
    # Leer la estructura del binario original
    print("Leyendo estructura del binario original...")
    with open(base_bin, 'rb') as f:
        # Header
        header = f.read(5)
        print(f"  Header: {header}")
        
        # Leer todos los valores de config (cada uno 2 bytes)
        all_values = []
        for i in range(100):
            b1, b2 = f.read(2)
            val = struct.unpack('<H', bytes([b1, b2]))[0]
            all_values.append(val)
            if val == 0 and i > 2:
                break
        
        print(f"  Valores leídos: {all_values[:15]}")
        
        # Leer todos los tensores del archivo original
        print("Leyendo tensores del archivo original...")
        tensors_original = []
        while True:
            try:
                # Leer longitud del nombre del tensor
                name_len_bytes = f.read(2)
                if len(name_len_bytes) < 2:
                    break
                name_len = struct.unpack('<H', name_len_bytes)[0]
                
                # Si name_len es muy grande, podría ser un valor de config
                if name_len > 2000:
                    break
                
                # Leer nombre del tensor
                if name_len > 0 and name_len < 500:
                    name = f.read(name_len).decode('utf-8').rstrip('\x00')
                else:
                    name = f.read(name_len).decode('latin-1').rstrip('\x00')
                
                # Leer dimensiones (4 valores de 2 bytes cada uno)
                d0 = struct.unpack('<H', f.read(2))[0]
                d1 = struct.unpack('<H', f.read(2))[0]
                d2 = struct.unpack('<H', f.read(2))[0]
                d3 = struct.unpack('<H', f.read(2))[0]
                
                # Calcular tamaño de datos
                if d0 and d1 and d2 and d3:
                    data_len = d0 * d1 * d2 * d3 * 4
                elif d0 and d1:
                    data_len = d0 * d1 * 4
                else:
                    data_len = 4
                
                # Leer datos del tensor
                tensor_data = f.read(data_len)
                
                if name:
                    print(f"  Tensor: {name} dims=({d0},{d1},{d2},{d3}) size={data_len/(1024*1024):.1f}MB")
                    tensors_original.append({
                        'name': name,
                        'dims': (d0, d1, d2, d3),
                        'data': tensor_data,
                        'size': data_len
                    })
            except Exception as e:
                print(f"  Fin o error al leer tensor: {e}")
                break
        
        total_tensors = len(tensors_original)
        print(f"  Total de tensores leídos: {total_tensors}")
    
    # Generar nuevo binario con la misma estructura
    print(f"Generando formato binario: {bin_path}")
    with open(bin_path, 'wb') as f:
        # Header
        f.write(header)
        
        # Config
        for val in all_values:
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
    
    print(f"  Binario: {os.path.getsize(bin_path) / (1024*1024):.1f} MB")
    print("  Conversión completada")

if __name__ == "__main__":
    main()