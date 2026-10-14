#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Start V1 Model Server (Port 8084) - OLD MODEL
# ═══════════════════════════════════════════════════════════

set -e

MODEL="models/euromillon-v1-gpt2.gguf"
HOST="127.0.0.1"
PORT=8084
THREADS=8

if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    echo "   Available: ls models/*.gguf"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  V1 MODEL SERVER (OLD - 3159 samples, 2000 steps)        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Threads: $THREADS"
echo ""
echo "API: http://$HOST:$PORT/v1/chat/completions"
echo ""
echo "Press Ctrl+C to stop"
echo ""

~/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  -c 2048 \
  -t "$THREADS" \
  --host "$HOST" \
  --port "$PORT" \
  --threads "$THREADS"