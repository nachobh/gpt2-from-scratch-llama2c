#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Start V2 Model Server (Port 8085) - NEWER BETTER MODEL
# ═══════════════════════════════════════════════════════════

set -e

MODEL="models/euromillon-v2-gpt2.gguf"
HOST="127.0.0.1"
PORT=8085
THREADS=8

if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    echo "   Run: python3 build-gguf-v2.py"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  V2 MODEL SERVER (NEW)                                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Threads: $THREADS"
echo ""
echo "API: http://$HOST:$PORT/v1/chat/completions"
echo ""
echo "To test: curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"1, 2, 3\"}], \"max_tokens\": 50}'"
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
