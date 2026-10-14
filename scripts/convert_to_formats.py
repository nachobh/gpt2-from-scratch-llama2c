#!/usr/bin/env python3
"""
Convierte el modelo entrenado a formatos .safetensors y .gguf.
"""
import os
import struct
import torch
import json
from pathlib import Path

# Importar modelo de llama2.c
LLAMA2C_DIR = Path(__file__).parent.parent / "llama2.c"
import sys
sys.path.insert(0, str(LLAMA2C_DIR))
from model import ModelArgs, Transformer

# Importar safetensors
try:
    import safetensors.torch as st
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("WARNING: safetensors not installed. Installing...")
    os.system("pip3 install safetensors -q")
    import safetensors.torch as st
    HAS_SAFETENSORS = True

def load_model(checkpoint_path):
    """Carga el modelo desde el checkpoint."""
    print(f"Cargando modelo desde: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extraer config - puede tener diferentes claves
    config_dict = checkpoint.get('config', {})
    print(f"Config keys: {list(config_dict.keys())}")
    
    # Extraer valores con fallbacks
    n_layer = config_dict.get('n_layer') or config_dict.get('n_layers') or 8
    n_head = config_dict.get('n_head') or config_dict.get('n_heads') or 8
    n_embd = config_dict.get('n_embd') or config_dict.get('dim') or 512
    vocab_size = checkpoint.get('dataset', {}).get('vocab_size') or 45
    
    print(f"Config: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}, vocab_size={vocab_size}")
    
    # Crear modelo con vocab_size correcto
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
    
    model = Transformer(config)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    # Override vocab_size from dataset info
    dataset_info = checkpoint.get('dataset', {})
    if 'vocab_size' in dataset_info:
        config.vocab_size = dataset_info['vocab_size']
        # Update output layer
        model.output = torch.nn.Linear(config.dim, config.vocab_size, bias=False)
        model.output.weight = torch.nn.Parameter(checkpoint['model_state']['output.weight'])
    
    return model, config, checkpoint

def convert_to_safetensors(model, config, output_path):
    """Convierte el modelo a formato .safetensors."""
    if not HAS_SAFETENSORS:
        print("ERROR: safetensors not available")
        return
    
    print(f"\nConvirtiendo a .safetensors: {output_path}")
    
    # Guardar estado del modelo
    state_dict = model.state_dict()
    st.save_file(state_dict, output_path)
    
    # Guardar metadata
    meta = {
        'config': {
            'n_layer': config.n_layers,
            'n_head': config.n_heads,
            'n_embd': config.dim,
            'vocab_size': config.vocab_size,
            'hidden_dim': config.hidden_dim,
            'max_seq_len': config.max_seq_len
        }
    }
    with open(output_path.replace('.safetensors', '.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"✓ .safetensors guardado: {output_path} ({size_mb:.1f} MB)")

def convert_to_gguf(model, config, output_path):
    """Convierte el modelo a formato .gguf (llama2.c format)."""
    print(f"\nConvirtiendo a .gguf: {output_path}")
    
    with open(output_path, 'wb') as f:
        # Header: magic bytes + config
        f.write(b'G2M2')  # Magic bytes
        total_params = sum(p.numel() for p in model.parameters())
        f.write(struct.pack('<I', total_params))
        f.write(struct.pack('<I', config.dim))
        f.write(struct.pack('<I', config.n_layers))
        f.write(struct.pack('<I', config.n_heads))
        f.write(struct.pack('<I', config.vocab_size))
        
        # Guardar cada tensor como float16
        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('<I', tensor.numel()))
            f.write(struct.pack('<I', len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack('<I', dim))
            # Convertir a float16 y escribir
            fp16_tensor = tensor.half().cpu().numpy()
            f.write(fp16_tensor.tobytes())
    
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"✓ .gguf guardado: {output_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to training checkpoint")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--format", choices=["all", "safetensors", "gguf"], default="all")
    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, config, checkpoint = load_model(checkpoint_path)
    
    base_name = Path(checkpoint_path).stem
    safetensors_path = os.path.join(output_dir, f"{base_name}.safetensors")
    gguf_path = os.path.join(output_dir, f"{base_name}.gguf")
    
    if args.format in ["all", "safetensors"]:
        convert_to_safetensors(model, config, safetensors_path)
    
    if args.format in ["all", "gguf"]:
        convert_to_gguf(model, config, gguf_path)
    
    print("\n✓ CONVERSIÓN COMPLETADA")
