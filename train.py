#!/usr/bin/env python3
"""
Entrenamiento de modelo GPT-2 tiny con llama2.c model.py
Entrena un modelo desde cero con texto de Euromillón.

USO:
    python3 train.py <dataset.txt> <output> [OPTIONS]
    (guarda <output>.pt y <output>.meta.json; NO un .bin)
    
OPCIONES:
    --n-layer N       Número de capas (default: 8)
    --n-head N        Número de heads (default: 8)
    --n-embd N        Tamaño embedding (default: 512)
    --n-step N        Número de pasos (default: 10000)
    --batch-size N    Batch size (default: 32)
    --seq-len N       Longitud secuencia (default: 128)
    --lr LR           Learning rate (default: 0.002)
    --warmup N        Warmup steps (default: 100)
    --threads N       Threads para carga de datos (default: 8)
    --resume PATH     Reanudar desde checkpoint (opcional)
"""

# Disable torch._dynamo to avoid triton compatibility issues
import os
os.environ['TORCH_DYNAMO_ENABLED'] = '0'

import sys
import os
import math
import time
import json
import argparse
from pathlib import Path

# Agregar llama2.c al path para importar model.py
LLAMA2C_DIR = Path(__file__).parent / "llama2.c"
sys.path.insert(0, str(LLAMA2C_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

# Importar modelo de llama2.c
from model import ModelArgs, precompute_freqs_cis, apply_rotary_emb, repeat_kv, RMSNorm, Attention, FeedForward, TransformerBlock, Transformer

# ── Dataset personalizado ────────────────────────────────────────

class TextDataset(Dataset):
    """Carga texto y crea chunks de secuencias."""
    
    def __init__(self, filepath, seq_len, vocab_size=32000):
        # Leer texto
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Crear vocabulario (mapeo carácter → índice)
        chars = sorted(list(set(text)))
        self.vocab_size = min(len(chars), vocab_size)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        
        # Convertir texto a índices
        self.data = []
        for char in text:
            if char in self.char_to_idx:
                self.data.append(self.char_to_idx[char])
        
        self.seq_len = seq_len
        self.n_samples = len(self.data) // seq_len
        
        print(f"  Vocabulario: {self.vocab_size} caracteres únicos")
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

# ── Configuración del modelo ────────────────────────────────────────

def create_model_config(args):
    """Crea configuración del modelo."""
    return ModelArgs(
        dim=args.n_embd,
        n_layers=args.n_layer,
        n_heads=args.n_head,
        n_kv_heads=args.n_head,  # Same as n_heads for simplicity
        hidden_dim=args.n_embd * 4,
        max_seq_len=args.seq_len,
        norm_eps=1e-5,
        dropout=0.1,
        vocab_size=45  # Nuestro vocabulario de 45 caracteres
    )

# ── Entrenamiento ────────────────────────────────────────────────

def train(args):
    """Entrena el modelo."""
    
    # Configurar
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print("EUROMILLÓN LAMA2C - ENTRENAMIENTO")
    print(f"{'='*60}")
    print(f"  Dispositivo: {device}")
    print(f"  Modelo: GPT-2 tiny")
    print(f"  Capas: {args.n_layer}")
    print(f"  Heads: {args.n_head}")
    print(f"  Embedding: {args.n_embd}")
    print(f"  Pasos: {args.n_step}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Secuencia: {args.seq_len}")
    print()
    
    # Crear dataset
    print("→ Cargando dataset...")
    dataset = TextDataset(args.dataset, args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.threads,
        pin_memory=True
    )
    print(f"  Dataset cargado: {len(dataloader)} batches")
    print()
    
    # Crear modelo
    print("→ Creando modelo...")
    config = create_model_config(args)
    # Ajustar vocab_size al del dataset real (45 caracteres únicos)
    config.vocab_size = dataset.vocab_size
    print(f"  Config vocab_size: {config.vocab_size}")
    model = Transformer(config)
    model = model.to(device)
    
    # Debug: verificar capas del modelo
    print(f"  Model output layer: {model.output}")
    for name, param in model.named_parameters():
        if 'output' in name:
            print(f"  Output layer: {name} -> {param.shape}")
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros totales: {total_params:,}")
    print(f"  Parámetros entrenables: {trainable_params:,}")
    print()
    
    # Optimizador
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )
    
    # Scheduler (warmup + cosine decay)
    warmup_steps = args.warmup
    total_steps = args.n_step
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Reanudar desde checkpoint si existe
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"→ Reanudando desde checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_step = checkpoint['step']
        print(f"  Inicio: paso {start_step}")
    print()
    
    # Loop de entrenamiento
    print("INICIANDO ENTRENAMIENTO")
    print(f"{'='*60}")
    
    model.train()
    total_loss = 0
    losses = []
    start_time = time.time()
    
    for step in range(start_step, total_steps):
        # Obtener batch
        x_batch, y_batch = next(iter(dataloader))
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Obtener logits del modelo
        x = x_batch  # (batch, seq_len)
        logits = model(x)  # (batch, 1, vocab_size) - solo último token
        
        # Debug: print shapes
        if step % 100 == 0:
            print(f"  DEBUG: x_batch shape: {x_batch.shape}")
            print(f"  DEBUG: logits shape: {logits.shape}")
            print(f"  DEBUG: y_batch shape: {y_batch.shape}")
        
        # Calcular pérdida - usar último token de cada secuencia como target
        # logits: (batch, 1, vocab_size), y_batch: (batch, seq_len)
        # Usamos y[:, -1] (último token de cada secuencia) como target
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_batch[:, -1]  # Solo el último token de cada secuencia
        )
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        clip_grad_norm_(model.parameters(), 1.0)
        
        # Actualizar parámetros
        optimizer.step()
        scheduler.step()
        
        # Logging
        total_loss += loss.item()
        losses.append(loss.item())
        
        if (step + 1) % 100 == 0 or step == 0:
            avg_loss = total_loss / 100
            elapsed = time.time() - start_time
            speed = 100 / elapsed if elapsed > 0 else 0
            lr = scheduler.get_last_lr()[0]
            
            print(f"  Paso {step+1}/{total_steps} | "
                  f"Pérdida: {avg_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Velocidad: {speed:.1f} steps/s | "
                  f"Tiempo: {elapsed:.1f}s")
            
            total_loss = 0
            losses = []
        
        # Guardar checkpoint cada 1000 pasos
        if (step + 1) % 1000 == 0:
            checkpoint = {
                'step': step + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'config': config.__dict__ if hasattr(config, '__dict__') else None
            }
            ckpt_path = f"{args.output}.checkpoint_{step+1}.pt"
            torch.save(checkpoint, ckpt_path)
            print(f"  ✓ Checkpoint guardado: {ckpt_path}")
    
    # Guardar modelo final
    print(f"\n{'='*60}")
    print("GUARDANDO MODELO")
    print(f"{'='*60}")
    
    # Guardar como checkpoint de PyTorch (formato nativo)
    model.eval()
    with torch.no_grad():
        state_dict = model.state_dict()
        
        # Guardar como checkpoint PyTorch
        checkpoint = {
            'model_state': state_dict,
            'config': config.__dict__ if hasattr(config, '__dict__') else None,
            'dataset': {
                'vocab_size': dataset.vocab_size,
                'chars': len(dataset.data)
            }
        }
        torch.save(checkpoint, f"{args.output}.pt")
    
    print(f"✓ Modelo guardado: {args.output}.pt")
    print(f"  Tamaño: {os.path.getsize(f'{args.output}.pt') / (1024*1024):.1f} MB")
    print(f"  Parámetros: {total_params:,}")
    
    # Guardar metadatos
    metadata = {
        'config': {
            'n_layer': args.n_layer,
            'n_head': args.n_head,
            'n_embd': args.n_embd,
            'batch_size': args.batch_size,
            'seq_len': args.seq_len,
            'lr': args.lr,
            'n_step': args.n_step,
            'warmup': args.warmup
        },
        'dataset': {
            'path': args.dataset,
            'vocab_size': dataset.vocab_size,
            'chars': len(dataset.data)
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
    
    meta_path = f"{args.output}.meta.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadatos guardados: {meta_path}")
    print(f"\n✓ ENTRENAMIENTO COMPLETADO")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo GPT-2 tiny")
    parser.add_argument("dataset", help="Ruta al archivo de texto de entrenamiento")
    parser.add_argument("output", help="Ruta base de salida (se guardan .pt y .meta.json)")
    parser.add_argument("--n-layer", type=int, default=8, help="Número de capas")
    parser.add_argument("--n-head", type=int, default=8, help="Número de heads")
    parser.add_argument("--n-embd", type=int, default=512, help="Tamaño embedding")
    parser.add_argument("--n-step", type=int, default=10000, help="Número de pasos")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Longitud secuencia")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup steps")
    parser.add_argument("--threads", type=int, default=8, help="Threads para carga de datos")
    parser.add_argument("--resume", type=str, help="Reanudar desde checkpoint")
    
    import struct
    args = parser.parse_args()
    
    train(args)
