#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Convert PyTorch model to GGUF format
# ═══════════════════════════════════════════════════════════

set -e

MODEL_NAME="${1:-euromillon-v2-gpt2}"
MODEL_PT="models/${MODEL_NAME}.pt"
OUTPUT_GGUF="models/${MODEL_NAME}.gguf"

if [ ! -f "$MODEL_PT" ]; then
    echo "❌ Model not found: $MODEL_PT"
    echo "   Run: bash 3a-phase1-train.sh"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  CONVERTING TO GGUF FORMAT                               ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Source: $MODEL_PT"
echo "Output: $OUTPUT_GGUF"
echo ""

# Convert using llama-convert-llama2c-to-ggml
~/llama.cpp/build/bin/llama-convert-llama2c-to-ggml \
  --llama2c-model "$MODEL_PT" \
  --llama2c-output-model "$OUTPUT_GGUF"

echo ""
echo "✓ Conversion complete: $OUTPUT_GGUF"
echo "  Size: $(du -h "$OUTPUT_GGUF" | cut -f1)"
