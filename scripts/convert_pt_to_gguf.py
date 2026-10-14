#!/usr/bin/env python3
"""Convierte el modelo SWE-Zero desde .pt a formato GGUF."""
import os, sys, subprocess, json

LLAMA2C_DIR = os.path.join(os.path.dirname(__file__), '..', 'llama2.c')
sys.path.insert(0, LLAMA2C_DIR)
import torch
import safetensors.torch as st

def main():
    # Usar el archivo .pt del modelo SWE-Zero
    pt_path = "models/euromillon-swe-zero.pt"
    gguf_path = "models/euromillon-swe-zero.gguf"
    
    # Convertir .safetensors a .pt primero
    safetensors_path = "models/euromillon-swe-zero.safetensors"
    if not os.path.exists(pt_path):
        print("Convirtiendo .safetensors a .pt...")
        state_dict = st.load_file(safetensors_path)
        torch.save(state_dict, pt_path)
        print(f"  .pt generado: {pt_path}")
    
    # Copiar el archivo .pt con el formato correcto para export.py
    # El export.py espera un archivo .pt con keys 'model_state' y 'config'
    print(f"Cargando modelo desde {pt_path}...")
    swere_dict = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    # Cargar config del modelo base
    base_pt = "models/euromillon-base-gpt2.pt"
    if os.path.exists(base_pt):
        base = torch.load(base_pt, map_location='cpu', weights_only=False)
        config = base.get('config', {})
    else:
        config = {
            'dim': 512,
            'n_layers': 8,
            'n_heads': 8,
            'n_kv_heads': 8,
            'vocab_size': 45,
            'hidden_dim': 2048,
            'multiple_of': 256,
            'norm_eps': 1e-5,
            'max_seq_len': 128,
            'dropout': 0.1
        }
    
    # Crear archivo .pt con el formato esperado por export.py
    print("Creando archivo .pt con formato correcto...")
    export_pt = "models/euromillon-swe-zero-export.pt"
    export_data = {
        'model': swere_dict,
        'model_args': config,
        'config': config
    }
    torch.save(export_data, export_pt)
    print(f"  Archivo .pt creado: {export_pt}")
    
    # Convertir a GGUF usando export.py
    print("Convirtiendo a GGUF usando export.py...")
    export_script = os.path.join(LLAMA2C_DIR, "export.py")
    
    result = subprocess.run(
        ['python3', export_script, '--checkpoint', export_pt, gguf_path],
        capture_output=True, text=True, timeout=300
    )
    
    if result.returncode == 0:
        print(f"  ✓ GGUF generado: {gguf_path}")
        size_mb = os.path.getsize(gguf_path) / (1024*1024)
        print(f"    Tamaño: {size_mb:.1f} MB")
        
        # Limpiar archivos temporales
        if os.path.exists(export_pt):
            os.remove(export_pt)
        
        return True
    else:
        print(f"  ✗ Error en conversión:")
        print(f"    {result.stderr[:500]}")
        
        # Intentar limpiar
        if os.path.exists(export_pt):
            os.remove(export_pt)
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)