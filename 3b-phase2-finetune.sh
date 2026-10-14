#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# FASE 2: FINETUNE CON SWE-zerO (318,115 trayectorias)
# ═══════════════════════════════════════════════════════════════
#
# Toma el modelo base entrenado con datos de Euromillón
# y lo finetunea con SWE-Zero para dar "inteligencia" general.
#
# SWE-Zero contiene trayectorias de resolución de problemas,
# lo que ayuda al modelo a generar combinaciones más "lógicas".
#
# ═══════════════════════════════════════════════════════════════

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  FASE 2: FINETUNE CON SWE-ZERO                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ── CONFIGURACIÓN ───────────────────────────────────────────
# Modelo base entrenado en fase 1
BASE_MODEL="models/euromillon-base-gpt2.bin"

# Dataset SWE-Zero - ajusta la ruta según tu descarga
SWE_ZERO_DIR="data/swe-zero"
SWE_ZERO_TRAIN="$SWE_ZERO_DIR/train.txt"

# Parámetros de finetune (menor LR para preservar conocimiento existente)
N_LAYER=8
N_HEAD=8
N_EMBD=512
N_STEPS=5000
BATCH_SIZE=16
SEQ_LEN=256
LEARNING_RATE="0.0005"  # Menor que fase 1
WARMUP_STEPS=50
THREADS=32

# Salida del finetune
OUTPUT_DIR="models"
OUTPUT_NAME="euromillon-swe-zero"

# ── Verificar archivos ──────────────────────────────────────
if [ ! -f "$BASE_MODEL" ]; then
    echo "❌ Error: Modelo base no encontrado: $BASE_MODEL"
    echo "   Ejecuta primero: bash 3a-phase1-train.sh"
    exit 1
fi

if [ ! -f "$SWE_ZERO_TRAIN" ]; then
    echo "⚠️  Dataset SWE-Zero no encontrado: $SWE_ZERO_TRAIN"
    echo "   Descargando desde HuggingFace..."
    mkdir -p "$SWE_ZERO_DIR"
    cd "$SWE_ZERO_DIR"
    
    # Descargar SWE-Zero desde HuggingFace
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download swe/swe-zero --include "train.txt" --local-dir .
    else
        # Alternativa con curl (ajustar URL según el dataset real)
        echo "   Nota: Instala huggingface-cli: pip install huggingface-cli"
        echo "   Descarga manual: https://huggingface.co/datasets/swe/swe-zero"
    fi
    cd ..
fi

echo "→ Modelo base: $BASE_MODEL"
echo "   Tamaño: $(du -h "$BASE_MODEL" | cut -f1)"
echo ""
echo "→ Dataset SWE-Zero: $SWE_ZERO_TRAIN"
echo "   Tamaño: $(du -h "$SWE_ZERO_TRAIN" | cut -f1)"
echo ""

# ── Ejecutar finetune FASE 2 ────────────────────────────────
echo "INICIANDO FASE 2: Finetune con SWE-Zero"
echo "  Modelo base: $BASE_MODEL"
echo "  Dataset: $SWE_ZERO_TRAIN"
echo "  Steps: $N_STEPS"
echo "  LR: $LEARNING_RATE (menor para preservar conocimiento)"
echo ""

cd llama2.c

# Ejecutar finetune (el modelo base se usa como punto de partida)
./run \
    -i "$SWE_ZERO_TRAIN" \
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
    -m "$BASE_MODEL"  # Cargar modelo base

echo ""
echo "✓ FASE 2 COMPLETADA"
echo "✓ Modelo finetuneado guardado: $OUTPUT_DIR/$OUTPUT_NAME.bin"
echo ""
echo "PRÓXIMO: Convertir a GGUF y generar combinaciones"