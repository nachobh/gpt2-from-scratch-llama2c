#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# FASE 1: Entrenar modelo BASE con datos de Euromillón
# ═══════════════════════════════════════════════════════════════

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  FASE 1: ENTRENAMIENTO BASE CON DATOS DE EUROMILLÓN      ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ── CONFIGURACIÓN ───────────────────────────────────────────
N_LAYER=8
N_HEAD=8
N_EMBD=512
N_STEPS=10000
BATCH_SIZE=32
SEQ_LEN=128
LEARNING_RATE="0.002"
WARMUP_STEPS=100
THREADS=8

# Rutas
DATASET="data/euromillon_real.txt"
OUTPUT_DIR="models"
OUTPUT_NAME="euromillon-base-gpt2"

# ── Verificar archivos ──────────────────────────────────────
if [ ! -f "$DATASET" ]; then
    echo "❌ Error: Dataset no encontrado: $DATASET"
    echo "   Ejecuta: python3 2a-extract-real-data.py"
    exit 1
fi

echo "→ Dataset: $DATASET"
echo "   Tamaño: $(du -h "$DATASET" | cut -f1)"
echo ""

# ── Ejecutar entrenamiento ──────────────────────────────────
echo ""
echo "INICIANDO FASE 1: Entrenamiento base"
echo "  Capas: $N_LAYER"
echo "  Heads: $N_HEAD"
echo "  Embedding: $N_EMBD"
echo "  Steps: $N_STEPS"
echo "  Batch: $BATCH_SIZE"
echo "  LR: $LEARNING_RATE"
echo "  Secuencia: $SEQ_LEN"
echo ""

python3 train.py \
    "$DATASET" \
    "$OUTPUT_DIR/$OUTPUT_NAME" \
    --n-layer $N_LAYER \
    --n-head $N_HEAD \
    --n-embd $N_EMBD \
    --n-step $N_STEPS \
    --batch-size $BATCH_SIZE \
    --seq-len $SEQ_LEN \
    --lr $LEARNING_RATE \
    --warmup $WARMUP_STEPS \
    --threads $THREADS

echo ""
echo "✓ FASE 1 COMPLETADA"
echo "✓ Modelo base guardado: $OUTPUT_DIR/$OUTPUT_NAME"
echo ""
echo "PRÓXIMAMENTE: FASE 2 - Finetune con SWE-Zero"