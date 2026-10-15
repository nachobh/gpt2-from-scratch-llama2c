#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Run Euromillon model with llama.cpp/llama-cli
# ═══════════════════════════════════════════════════════════

set -e

MODEL_PATH="${1:-models/euromillon-v2-gpt2.gguf}"
TEMPERATURE="${2:-0.7}"
PREDICTS="${3:-50}"
PROMPT="${4:-Genera una combinacion de Euromillon:}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "   Run: bash 4-convert.sh"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  LLAMA.CPP - EUROMILLON MODEL GENERATOR                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL_PATH"
echo "Temperature: $TEMPERATURE"
echo "Predicts: $PREDICTS"
echo "Prompt: $PROMPT"
echo ""

# Run with llama-cli (llama.cpp)
# -t = number of THREADS (not temperature); temperature is --temp
~/llama.cpp/build/bin/llama-cli \
  -m "$MODEL_PATH" \
  -n "$PREDICTS" \
  -c 2048 \
  -t 4 \
  --temp "$TEMPERATURE" \
  -s 42 \
  -p "$PROMPT"
