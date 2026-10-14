#!/usr/bin/env python3
"""
Convierte el modelo SWE-Zero a formato GGUF usando el formato binario correcto.
"""
import os
import sys
import struct
import json
import torch
import safetensors.torch as st

LLAMA2C_DIR = os.path.join(os.path.dirname(__file__), '..', 'llama2.c')
sys.path.insert(0, LLAMA2C_DIR)
from model import ModelArgs

def convert_to_gguf(safetensors_path, gguf_path):
    print(f"Convirtiendo {safetensors_path} → {gguf_path}")
    
    # Cargar el modelo
    print("Cargando modelo...")
    state_dict = st.load_file(safetensors_path)
    
    # Cargar config desde metadatos
    meta_path = safetensors_path + '.meta.json'
    config_dict = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            config_dict = meta.get('config', {})
    
    # Extraer config
    n_layer = config_dict.get('n_layer', 8)
    n_head = config_dict.get('n_head', 8)
    n_embd = config_dict.get('n_embd', 512)
    vocab_size = config_dict.get('vocab_size', 45)
    
    # Crear config para el formato binario
    model_args = {
        'dim': n_embd,
        'n_layers': n_layer,
        'n_heads': n_head,
        'n_kv_heads': n_head,
        'hidden_dim': n_embd * 4,
        'max_seq_len': 128,
        'norm_eps': 1e-5,
        'dropout': 0.1,
        'vocab_size': vocab_size
    }
    
    config_for_binary = {
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'n_kv_heads': n_head,
        'hidden_dim': n_embd * 4,
        'max_seq_len': 128,
        'norm_eps': 1e-5,
        'vocab_size': vocab_size,
        'model_args': model_args
    }
    
    # Generar el formato binario correcto
    binary_path = safetensors_path.replace('.safetensors', '.bin')
    print(f"Generando formato binario: {binary_path}")
    
    # Header del formato llama2.c
    header = b'G2M2\x00'
    
    # Escribir header
    with open(binary_path, 'wb') as f:
        f.write(header)
        
        # Escribir config como JSON
        config_json = json.dumps(config_for_binary).encode('utf-8')
        f.write(struct.pack('<I', len(config_json)))
        f.write(config_json)
        
        # Escribir cada tensor
        for key, tensor in state_dict.items():
            # Nombre del tensor
            name_bytes = key.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            
            # Datos del tensor
            tensor_float32 = tensor.float()
            data = tensor_float32.cpu().numpy().tobytes('C')
            f.write(data)
    
    print(f"  Binario generado: {os.path.getsize(binary_path) / (1024*1024):.1f} MB")
    
    # Convertir a GGUF usando llama-convert-llama2c-to-ggml
    print("Convirtiendo a GGUF...")
    import subprocess
    converter = '/usr/local/bin/llama-convert-llama2c-to-ggml'
    
    if not os.path.exists(converter):
        print(f"  ✗ Conversor no encontrado: {converter}")
        return False
    
    result = subprocess.run(
        [converter, '--llama2c-model', binary_path, 
         '--copy-vocab-from-model', '~/models/gemma4-v2-Q8_0.gguf',
         '--llama2c-output-model', gguf_path],
        capture_output=True, text=True, timeout=300
    )
    
    if result.returncode == 0:
        print(f"  ✓ GGUF generado: {gguf_path}")
        size_mb = os.path.getsize(gguf_path) / (1024*1024)
        print(f"    Tamaño: {size_mb:.1f} MB")
        
        # Limpiar archivo temporal
        if os.path.exists(binary_path):
            os.remove(binary_path)
        
        return True
    else:
        print(f"  ✗ Error en conversión:")
        print(f"    {result.stderr[:500]}")
        
        # Intentar limpiar
        if os.path.exists(binary_path):
            os.remove(binary_path)
        
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convertir SWE-Zero a GGUF")
    parser.add_argument("--model-path", required=True, help="Ruta al modelo .safetensors")
    parser.add_argument("--output-path", required=True, help="Ruta de salida .gguf")
    args = parser.parse_args()
    
    success = convert_to_gguf(args.model_path, args.output_path)
    sys.exit(0 if success else 1)