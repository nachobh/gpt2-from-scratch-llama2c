#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Run Euromillon model with llama.c/runq
# ═══════════════════════════════════════════════════════════

set -e

MODEL_PATH="${1:-models/euromillon-v2-gpt2.gguf}"
TEMPERATURE="${2:-0.7}"
SEQUENCE="${3:-Genera una combinacion de Euromillon:}"
PREDICTS="${4:-50}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "   Run: bash 4-convert.sh"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  LLAMA.C - EUROMILLON MODEL GENERATOR                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL_PATH"
echo "Temperature: $TEMPERATURE"
echo "Predicts: $PREDICTS"
echo ""

# Run with runq (quantized version)
~/euromillon-llama2c/llama2.c/runq "$MODEL_PATH" \
  -t "$TEMPERATURE" \
  -s 42 \
  -n "$PREDICTS" \
  -i "$SEQUENCE" \
  -m chat
