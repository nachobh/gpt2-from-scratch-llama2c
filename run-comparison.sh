#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Compare V1 and V2 models - Direct Python Inference
# Uses PT models directly (no GGUF dependency)
# ═══════════════════════════════════════════════════════════

set -e
cd ~/euromillon-llama2c

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  EUROMILLON MODEL COMPARISON (V1 vs V2)                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "V1: 3,159 samples, 2000 steps, 65 MB GGUF (original)"
echo "V2: 11,159 samples, 500 steps, 129 MB GGUF (merged)"
echo ""

# Test prompts
PROMPTS=("1, 2, 3, 4, 5," "10, 20, 30," "1, 3, 5, 7, 9,")

for PROMPT in "${PROMPTS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "PROMPT: $PROMPT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    for MODEL in "v1" "v2"; do
        if [ "$MODEL" = "v1" ]; then
            FILE="models/euromillon-base-gpt2.pt"
            DESC="V1 (3,159 samples, 2000 steps)"
        else
            FILE="models/euromillon-v2-gpt2.pt"
            DESC="V2 (11,159 samples, 500 steps)"
        fi
        
        echo ""
        echo "[$DESC] Generating..."
        python3 -c "
import torch, json, sys
sys.path.insert(0, '.')
exec(open('compare.py').read().replace('MODEL_FILE', '$FILE').replace('PROMPT', '$PROMPT'))
" 2>/dev/null
    done
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SERVERS (if running):"
echo "  V1: http://127.0.0.1:8084/v1/chat/completions"
echo "  V2: http://127.0.0.1:8085/v1/chat/completions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
