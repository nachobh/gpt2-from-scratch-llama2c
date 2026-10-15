#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Test the V1 model with different prompts
# ═══════════════════════════════════════════════════════════

MODEL="models/euromillon-base-gpt2.gguf"

if [ ! -f "$MODEL" ]; then
    echo "❌ Model not found: $MODEL"
    exit 1
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  EUROMILLON AI - MODEL V1 TEST                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Testing model: $MODEL"
echo ""

# Test 1: Basic generation
# -t = threads (not temperature); temperature is --temp
echo "--- TEST 1: Basic generation ---"
~/llama.cpp/build/bin/llama-cli \
  -m "$MODEL" \
  -n 50 \
  -c 2048 \
  -t 4 \
  --temp 0.7 \
  -s 42 \
  -p "Genera una combinacion de Euromillon:"
echo ""
echo ""

# Test 2: With context
echo "--- TEST 2: With context ---"
~/llama.cpp/build/bin/llama-cli \
  -m "$MODEL" \
  -n 50 \
  -c 2048 \
  -t 4 \
  --temp 0.7 \
  -s 42 \
  -p "La combinacion del dia de hoy:"
echo ""
echo ""

# Test 3: With statistics
echo "--- TEST 3: With statistics ---"
~/llama.cpp/build/bin/llama-cli \
  -m "$MODEL" \
  -n 50 \
  -c 2048 \
  -t 4 \
  --temp 0.7 \
  -s 42 \
  -p "Segun las estadisticas, una buena combinacion es:"
