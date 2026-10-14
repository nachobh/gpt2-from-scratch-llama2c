#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Run Euromillon model as REST API with llama.cpp/llama-server
# ═══════════════════════════════════════════════════════════

set -e

MODEL_PATH="${1:-models/euromillon-v2-gpt2.gguf}"
HOST="${2:-127.0.0.1}"
PORT="${3:-8084}"
THREADS="${4:-8}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Model not found: $MODEL_PATH"
    echo "   Run: bash 4-convert.sh"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  LLAMA.CPP SERVER - EUROMILLON API                       ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Threads: $THREADS"
echo ""
echo "API: http://$HOST:$PORT/v1/chat/completions"
echo ""

# Run llama-server
~/llama.cpp/build/bin/llama-server \
  -m "$MODEL_PATH" \
  -c 2048 \
  -t "$THREADS" \
  --host "$HOST" \
  --port "$PORT" \
  --threads "$THREADS"
