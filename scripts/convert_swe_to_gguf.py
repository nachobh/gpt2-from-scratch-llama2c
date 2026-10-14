#!/usr/bin/env python3
"""Convierte el modelo SWE-Zero a formato GGUF."""
import os, sys, struct, subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llama2.c'))
import torch
import safetensors.torch as st

def main():
    safetensors_path = "models/euromillon-swe-zero.safetensors"
    gguf_path = "models/euromillon-swe-zero.gguf"
    binary_path = "models/euromillon-swe-zero.bin"
    base_binary = "models/euromillon-base-gpt2"
    
    print(f"Convirtiendo {safetensors_path} -> {gguf_path}")
    
    # Cargar modelo SWE-Zero
    print("Cargando modelo SWE-Zero...")
    state_dict = st.load_file(safetensors_path)
    
    # Leer el archivo binario original para obtener la estructura exacta
    print("Leyendo estructura del binario original...")
    with open(base_binary, 'rb') as f:
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
        
        # El primer valor (124) es la longitud del nombre del primer tensor
        # Los siguientes son: n_layer, n_head, n_kv_heads, hidden_dim, max_seq_len, vocab_size
        # Valores típicos: 8, 8, 8, 2048, 128, 45
        
        # Reconstruir config con valores correctos
        n_layer = 8
        n_head = 8
        n_embd = 512
        vocab_size = 45
        max_seq_len = 128
        
        # Crear nueva config
        new_config = [
            1,  # count
            n_layer,
            n_head,
            n_head,  # n_kv_heads
            n_embd * 4,  # hidden_dim
            max_seq_len,
            vocab_size,
        ]
        
        # Rellenar hasta el número esperado de valores
        while len(new_config) < 20:
            new_config.append(0)
        
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
    print(f"Generando formato binario: {binary_path}")
    with open(binary_path, 'wb') as f:
        # Header
        f.write(header)
        
        # Config
        for val in new_config:
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
    
    # Convertir a GGUF
    print("Convirtiendo a GGUF...")
    converter = '/usr/local/bin/llama-convert-llama2c-to-ggml'
    result = subprocess.run(
        [converter, '--llama2c-model', binary_path,
         '--copy-vocab-from-model', '~/models/gemma4-v2-Q8_0.gguf',
         '--llama2c-output-model', gguf_path],
        capture_output=True, text=True, timeout=300
    )
    
    if result.returncode == 0:
        print(f"  ✓ GGUF: {gguf_path}")
        print(f"    Tamaño: {os.path.getsize(gguf_path) / (1024*1024):.1f} MB")
        os.remove(binary_path)
        return True
    else:
        print(f"  ✗ Error: {result.stderr[:300]}")
        if os.path.exists(binary_path):
            os.remove(binary_path)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)