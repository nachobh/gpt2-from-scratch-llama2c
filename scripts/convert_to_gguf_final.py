#!/usr/bin/env python3
"""Convierte el modelo SWE-Zero a formato GGUF usando el mismo método que el modelo base."""
import os, sys, subprocess, json, struct

LLAMA2C_DIR = os.path.join(os.path.dirname(__file__), '..', 'llama2.c')
sys.path.insert(0, LLAMA2C_DIR)
import torch
import safetensors.torch as st

def main():
    pt_path = "models/euromillon-swe-zero.pt"
    gguf_path = "models/euromillon-swe-zero.gguf"
    
    # Cargar el modelo desde .pt
    print(f"Cargando modelo desde {pt_path}...")
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    # Extraer config y model_state
    model_state = checkpoint.get('model_state', checkpoint.get('model', {}))
    config = checkpoint.get('config', {})
    
    # Crear archivo .pt con el formato que espera export.py
    export_pt = "models/euromillon-swe-zero-export.pt"
    export_data = {
        'model': model_state,
        'model_args': config,
        'config': config
    }
    torch.save(export_data, export_pt)
    print(f"Archivo .pt creado: {export_pt}")
    
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