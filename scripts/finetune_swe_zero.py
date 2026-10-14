#!/usr/bin/env python3
"""
Finetune del modelo Euromillón con SWE-Zero openhands-trajectories.
Adapta el modelo para entender contextos de ingeniería de software.
"""
import os
import sys
import time
import json
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

# Importar modelo de llama2.c
LLAMA2C_DIR = os.path.join(os.path.dirname(__file__), '..', 'llama2.c')
sys.path.insert(0, LLAMA2C_DIR)
from model import ModelArgs, Transformer

# Importar safetensors
import safetensors.torch as st

def load_safetensors_model(safetensors_path):
    """Carga el modelo desde archivo .safetensors."""
    print(f"Cargando modelo desde .safetensors: {safetensors_path}")
    state_dict = st.load_file(safetensors_path)
    
    # Cargar config desde JSON si existe
    json_path = safetensors_path.replace('.safetensors', '.json')
    config_dict = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            config_dict = json.load(f).get('config', {})
    
    # Crear modelo
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
    
    model = Transformer(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"  Parámetros cargados: {sum(p.numel() for p in model.parameters()):,}")
    return model, config

class SWEDataset(Dataset):
    """Dataset simplificado de SWE-Zero para finetune."""
    
    def __init__(self, dataset_path, seq_len, max_samples=None):
        from datasets import load_dataset
        
        print(f"Cargando dataset: {dataset_path}")
        ds = load_dataset("parquet", data_files=os.path.join(dataset_path, "data", "train-*.parquet"), split="train")
        
        # Limitar muestras para entrenamiento rápido
        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples)))
        
        print(f"  Muestras disponibles: {len(ds)}")
        
        # Extraer texto de cada trayectoria
        self.texts = []
        for row in ds:
            # Construir texto plano de la trayectoria
            text_parts = []
            for turn in row.get('trajectory', []):
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                if content and len(content) > 10:  # Ignorar turns vacíos
                    text_parts.append(f"[{role}] {content}")
            
            if text_parts:
                self.texts.append(" ".join(text_parts))
        
        # Crear vocabulario (usar el del modelo existente)
        all_text = " ".join(self.texts)
        chars = sorted(list(set(all_text)))
        # Limitar a 45 caracteres para compatibilidad
        chars = chars[:45]
        self.vocab_size = len(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        
        # Convertir textos a índices
        self.data = []
        for text in self.texts:
            indices = []
            for char in text:
                if char in self.char_to_idx:
                    indices.append(self.char_to_idx[char])
                else:
                    indices.append(0)  # Padding
            self.data.extend(indices)
        
        self.seq_len = seq_len
        self.n_samples = len(self.data) // seq_len
        
        print(f"  Textos útiles: {len(self.texts)}")
        print(f"  Vocabulario: {self.vocab_size} caracteres")
        print(f"  Total de caracteres: {len(self.data)}")
        print(f"  Secuencias: {self.n_samples}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = torch.tensor(self.data[start:end], dtype=torch.long)
        y = torch.tensor(self.data[start+1:end+1], dtype=torch.long)
        return x, y

def finetune(args):
    """Finetune el modelo con SWE-Zero."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print("FINETUNE CON SWE-ZERO")
    print(f"{'='*60}")
    print(f"  Dispositivo: {device}")
    print(f"  Modelo: Euromillón base")
    print(f"  Dataset: SWE-Zero openhands-trajectories")
    print(f"  Pasos: {args.n_step}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Secuencia: {args.seq_len}")
    print()
    
    # Cargar modelo
    model, config = load_safetensors_model(args.model_path)
    model = model.to(device)
    
    # Cargar dataset
    print("→ Cargando dataset SWE-Zero...")
    dataset = SWEDataset(
        args.dataset_path,
        args.seq_len,
        max_samples=args.max_samples
    )
    
    # Actualizar vocab_size del modelo
    if dataset.vocab_size != config.vocab_size:
        print(f"  Actualizando vocab_size: {config.vocab_size} → {dataset.vocab_size}")
        config.vocab_size = dataset.vocab_size
        model.output = torch.nn.Linear(config.dim, dataset.vocab_size, bias=False)
    
    # Recargar pesos de embedding y output
    state_dict = st.load_file(args.model_path)
    # Mover tensores al dispositivo correcto antes de cargar
    for key in ['tok_embeddings.weight', 'output.weight']:
        if key in state_dict:
            state_dict[key] = state_dict[key].to(device)
    
    model.load_state_dict(state_dict, strict=False)
    
    # CRUCIAL: Mover TODO el modelo al dispositivo después de cargar pesos
    model = model.to(device)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"  Dataset cargado: {len(dataloader)} batches")
    print()
    
    # Optimizador
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    # Scheduler
    warmup_steps = args.warmup
    total_steps = args.n_step
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Entrenamiento
    print("INICIANDO FINETUNE")
    print(f"{'='*60}")
    
    model.train()
    total_loss = 0
    losses = []
    start_time = time.time()
    
    for step in range(total_steps):
        x_batch, y_batch = next(iter(dataloader))
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        x = x_batch
        logits = model(x)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_batch[:, -1]
        )
        
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        losses.append(loss.item())
        
        if (step + 1) % 50 == 0 or step == 0:
            avg_loss = total_loss / max(len(losses), 1)
            elapsed = time.time() - start_time
            speed = max((step + 1) / elapsed, 0) if elapsed > 0 else 0
            lr = scheduler.get_last_lr()[0]
            
            print(f"  Paso {step+1}/{total_steps} | "
                  f"Pérdida: {avg_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Velocidad: {speed:.1f} steps/s | "
                  f"Tiempo: {elapsed:.1f}s")
            
            total_loss = 0
            losses = []
    
    # Guardar modelo final
    print(f"\n{'='*60}")
    print("GUARDANDO MODELO FINA-TUNEADO")
    print(f"{'='*60}")
    
    model.eval()
    state_dict = model.state_dict()
    
    # Guardar como .safetensors
    output_path = args.output_path
    
    # Usar save_model para manejar tensores compartidos automáticamente (weight tying)
    # save_model mueve todo al CPU internamente
    st.save_model(model, output_path)
    
    size_mb = os.path.getsize(output_path) / (1024*1024)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ .safetensors guardado: {output_path}")
    print(f"  Tamaño: {size_mb:.1f} MB")
    print(f"  Parámetros: {total_params:,}")
    
    # Guardar metadatos
    metadata = {
        'config': {
            'n_layer': config.n_layers,
            'n_head': config.n_heads,
            'n_embd': config.dim,
            'vocab_size': config.vocab_size,
            'batch_size': args.batch_size,
            'seq_len': args.seq_len,
            'lr': args.lr,
            'n_step': args.n_step,
            'warmup': args.warmup,
            'max_samples': args.max_samples
        },
        'dataset': {
            'path': args.dataset_path,
            'name': 'SWE-Zero-openhands-trajectories',
            'samples_used': args.max_samples or len(dataset)
        },
        'model': {
            'total_params': total_params,
            'device': device
        },
        'training': {
            'final_loss': sum(losses)/len(losses) if losses else 0,
            'total_time': time.time() - start_time
        }
    }
    
    meta_path = f"{output_path}.meta.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadatos guardados: {meta_path}")
    print(f"\n✓ FINETUNE COMPLETADO")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finetune con SWE-Zero")
    parser.add_argument("--model-path", required=True, help="Ruta al modelo .safetensors")
    parser.add_argument("--dataset-path", required=True, help="Ruta al dataset SWE-Zero")
    parser.add_argument("--output-path", required=True, help="Ruta de salida")
    parser.add_argument("--n-step", type=int, default=100, help="Número de pasos")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Longitud secuencia")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    parser.add_argument("--max-samples", type=int, default=1000, help="Máximo de muestras")
    args = parser.parse_args()
    
    finetune(args)
