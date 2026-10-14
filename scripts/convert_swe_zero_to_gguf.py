#!/usr/bin/env python3
"""
Convierte el modelo SWE-Zero finetuneado a formato GGUF.
"""
import os
import sys
import json
import torch
import safetensors.torch as st

# Importar modelo de llama2.c
LLAMA2C_DIR = os.path.join(os.path.dirname(__file__), '..', 'llama2.c')
sys.path.insert(0, LLAMA2C_DIR)
from model import ModelArgs, Transformer

# Importar conversor GGUF
try:
    from llama2.c import run
except ImportError:
    # Intentar importar el módulo de conversión
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("export", os.path.join(LLAMA2C_DIR, "export.py"))
        export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export)
    except Exception:
        pass

def convert_to_gguf(safetensors_path, gguf_path):
    """Convierte el modelo .safetensors a .gguf usando el conversor de llama2.c."""
    print(f"Convirtiendo {safetensors_path} → {gguf_path}")
    
    # Cargar el modelo
    print("Cargando modelo...")
    state_dict = st.load_file(safetensors_path)
    
    # Cargar config desde JSON si existe
    json_path = safetensors_path.replace('.safetensors', '.json')
    config_dict = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            config_dict = json.load(f).get('config', {})
    
    # Obtener config desde metadatos si existen
    meta_path = safetensors_path + '.meta.json'
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            config_dict = meta.get('config', config_dict)
    
    # Crear config
    n_layer = config_dict.get('n_layer', 8)
    n_head = config_dict.get('n_head', 8)
    n_embd = config_dict.get('n_embd', 512)
    vocab_size = config_dict.get('vocab_size', 45)
    
    config = ModelArgs(
        dim=n_embd,
        n_layers=n_layer,
        n_heads=n_head,
        n_kv_heads=n_head,
        hidden_dim=n_embd * 4,
        max_seq_len=128,
        norm_eps=1e-5,
        dropout=0.1,
        vocab_size=vocab_size
    )
    
    # Crear y cargar modelo
    model = Transformer(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"  Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Guardar como formato binario temporal que el conversor GGUF puede leer
    temp_binary_path = safetensors_path.replace('.safetensors', '.bin')
    print(f"Guardando formato binario temporal: {temp_binary_path}")
    
    # Guardar en formato que el conversor de llama2.c puede usar
    # El conversor llama2.c/export.py puede leer .pt o formato binario
    torch.save({
        'model_state': state_dict,
        'config': {
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'vocab_size': vocab_size,
            'batch_size': config_dict.get('batch_size', 32),
            'seq_len': config_dict.get('seq_len', 128),
            'lr': config_dict.get('lr', 0.0005),
            'n_step': config_dict.get('n_step', 50),
            'warmup': config_dict.get('warmup', 5),
            'max_samples': config_dict.get('max_samples', 500)
        }
    }, temp_binary_path)
    
    # Intentar convertir usando el export.py de llama2.c
    print("Intentando conversión a GGUF...")
    try:
        # Usar el módulo export.py directamente
        import importlib.util
        export_path = os.path.join(LLAMA2C_DIR, "export.py")
        spec = importlib.util.spec_from_file_location("export", export_path)
        export_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export_module)
        
        # Llamar a la función de exportación
        # export.py espera: --checkpoint <modelo> <salida>
        # Necesitamos pasar el modelo en formato que pueda leer
        print(f"  Usando export.py: {export_path}")
        
        # Guardar como .pt para que export.py lo lea
        pt_path = gguf_path.replace('.gguf', '.pt')
        torch.save({
            'model_state': state_dict,
            'config': {
                'n_layer': n_layer,
                'n_head': n_head,
                'n_embd': n_embd,
                'vocab_size': vocab_size
            }
        }, pt_path)
        
        # Llamar a export.py
        print(f"  Ejecutando: python3 {export_path} --checkpoint {pt_path} {gguf_path}")
        import subprocess
        result = subprocess.run(
            ['python3', export_path, '--checkpoint', pt_path, gguf_path],
            capture_output=True, text=True, timeout=300
        )
        
        if result.returncode == 0:
            print(f"  ✓ GGUF generado: {gguf_path}")
            size_mb = os.path.getsize(gguf_path) / (1024*1024)
            print(f"    Tamaño: {size_mb:.1f} MB")
        else:
            print(f"  ✗ Error en conversión: {result.stderr}")
            print(f"  Intentando método alternativo...")
            # Método alternativo: usar llama-convert-llama2c-to-ggml si está disponible
            import shutil
            converter = shutil.which('llama-convert-llama2c-to-ggml')
            if converter:
                print(f"  Usando {converter}")
                result = subprocess.run(
                    [converter, '--llama2c-model', temp_binary_path, '--copy-vocab-from-model',
                     '~/models/gemma4-v2-Q8_0.gguf', '--llama2c-output-model', gguf_path],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    print(f"  ✓ GGUF generado: {gguf_path}")
                    size_mb = os.path.getsize(gguf_path) / (1024*1024)
                    print(f"    Tamaño: {size_mb:.1f} MB")
                else:
                    print(f"  ✗ Error: {result.stderr}")
            else:
                print(f"  ✗ Conversor no encontrado")
                
    except Exception as e:
        print(f"  ✗ Error en conversión: {e}")
    
    # Limpiar archivos temporales
    if os.path.exists(temp_binary_path):
        os.remove(temp_binary_path)
    if os.path.exists(pt_path):
        os.remove(pt_path)
    
    print("✓ Conversión completada")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convertir SWE-Zero a GGUF")
    parser.add_argument("--model-path", required=True, help="Ruta al modelo .safetensors")
    parser.add_argument("--output-path", required=True, help="Ruta de salida .gguf")
    args = parser.parse_args()
    
    convert_to_gguf(args.model_path, args.output_path)
