#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# PASO 3: Entrenar modelo GPT-2 tiny con llama2.c
# ═══════════════════════════════════════════════════════════════
#
# Configura los hiperparámetros según tu hardware:
# - Para CPU rápido: n_layer=8, n_head=8, n_embd=512 (GPT-2 tiny)
# - Para CPU lento: n_layer=6, n_head=6, n_embd=256 (más pequeño)
# - Para más calidad: n_layer=12, n_head=12, n_embd=768 (GPT-2 medium)
#
# TIEMPO ESTIMADO (Ryzen 32 núcleos):
# - GPT-2 tiny: 20-40 minutos por 1000 steps
# - GPT-2 medium: 1-2 horas por 1000 steps
#
# ═══════════════════════════════════════════════════════════════

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  PASO 3: Entrenar modelo GPT-2                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ── CONFIGURACIÓN ───────────────────────────────────────────
# Model size: tiny (117M), medium (354M), large (774M)
MODEL_SIZE="tiny"
# Number of training steps (adjust based on dataset size)
N_STEPS=5000
# Batch size (memory dependent)
BATCH_SIZE=64
# Sequence length (context window)
SEQ_LEN=256
# Learning rate
LEARNING_RATE="0.002"
# Number of threads (use all CPU cores)
THREADS=32
# Number of warmup steps
WARMUP_STEPS=100

# ── Rutas ───────────────────────────────────────────────────
LLAMA2C_DIR="llama2.c"
DATASET="data/euromillon_train.txt"
OUTPUT_DIR="models"
OUTPUT_NAME="euromillon-gpt2-${MODEL_SIZE}"

# ── Verificar archivos ──────────────────────────────────────
if [ ! -d "$LLAMA2C_DIR" ]; then
    echo "❌ Error: No se encontró $LLAMA2C_DIR"
    echo "   Ejecuta primero: ./1-clone.sh"
    exit 1
fi

if [ ! -f "$DATASET" ]; then
    echo "❌ Error: No se encontró el dataset $DATASET"
    echo "   Ejecuta primero: python3 2-generate-data.py"
    exit 1
fi

# ── Configurar hiperparámetros según modelo ─────────────────
case "$MODEL_SIZE" in
    tiny)
        N_LAYER=8
        N_HEAD=8
        N_EMBD=512
        echo "→ Modelo: GPT-2 tiny (~117M parámetros)"
        ;;
    medium)
        N_LAYER=12
        N_HEAD=12
        N_EMBD=768
        echo "→ Modelo: GPT-2 medium (~354M parámetros)"
        ;;
    large)
        N_LAYER=24
        N_HEAD=16
        N_EMBD=1024
        echo "→ Modelo: GPT-2 large (~774M parámetros)"
        ;;
    *)
        echo "❌ Modelo desconocido: $MODEL_SIZE"
        echo "   Usar: tiny, medium, o large"
        exit 1
        ;;
esac

# ── Crear directorio de salida ──────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── Verificar compilación ───────────────────────────────────
if [ ! -f "$LLAMA2C_DIR/bin/train" ]; then
    echo "→ Compilando llama2.c..."
    cd "$LLAMA2C_DIR"
    make clean
    make
    cd ..
fi

# ── Ejecutar entrenamiento ──────────────────────────────────
echo ""
echo "Iniciando entrenamiento..."
echo "  Dataset: $DATASET"
echo "  Steps: $N_STEPS"
echo "  Layers: $N_LAYER"
echo "  Heads: $N_HEAD"
echo "  Embedding: $N_EMBD"
echo "  Batch size: $BATCH_SIZE"
echo "  Threads: $THREADS"
echo ""

cd "$LLAMA2C_DIR"

./bin/train \
    -i "$DATASET" \
    -o "$OUTPUT_DIR/$OUTPUT_NAME.bin" \
    -n $N_STEPS \
    -b $BATCH_SIZE \
    -s $SEQ_LEN \
    -l $LEARNING_RATE \
    -t $THREADS \
    -w $WARMUP_STEPS \
    -a $N_LAYER \
    -h $N_HEAD \
    -e $N_EMBD \
    -q 0 2>&1

echo ""
echo "✓ Entrenamiento completado"
echo "✓ Modelo guardado: $OUTPUT_DIR/$OUTPUT_NAME.bin"
