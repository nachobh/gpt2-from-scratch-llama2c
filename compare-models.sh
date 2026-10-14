#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Compare V1 and V2 models using direct Python inference
# ═══════════════════════════════════════════════════════════

set -e

cd ~/euromillon-llama2c

# Prompts to test
PROMPTS=(
    "1, 2, 3, 4, 5,"
    "10, 20, 30,"
    "1, 3, 5, 7, 9,"
)

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  COMPARING EUROMILLON MODELS (V1 vs V2)                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "V1: 3,159 samples, 2000 steps, 65 MB GGUF"
echo "V2: 11,159 samples, 500 steps, 128 MB GGUF"
echo ""
echo "Testing with direct Python inference..."
echo ""

for PROMPT in "${PROMPTS[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "PROMPT: $PROMPT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    for MODEL in "v1" "v2"; do
        if [ "$MODEL" = "v1" ]; then
            MODEL_FILE="models/euromillon-base-gpt2.pt"
            VERSION="V1 (3,159 samples, 2000 steps)"
        else
            MODEL_FILE="models/euromillon-v2-gpt2.pt"
            VERSION="V2 (11,159 samples, 500 steps)"
        fi
        
        echo "[$VERSION] Generating..."
        python3 -c "
import torch
import json

# Load model
weights = torch.load('$MODEL_FILE', weights_only=False)
model_state = weights['model_state']
config = weights['config']
dataset = weights['dataset']

# Extract config
vocab_size = dataset['vocab_size']
n_layers = config['n_layers']
n_heads = config['n_heads']
dim = config['dim']
hidden_dim = config['hidden_dim']
max_seq_len = config['max_seq_len']
norm_eps = config['norm_eps']

# Prepare tokens from prompt
prompt = '$PROMPT'
tokens = [ord(c) for c in prompt]
n_tokens = len(tokens)

# Build model
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x, start_pos):
        bsz, seq_len, _ = x.shape
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Simple attention (no masking for generation)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

# Load weights
model = nn.Module()
model.tok_embeddings = nn.Embedding(vocab_size, dim)
model.norm = RMSNorm(dim, float(norm_eps))
model.output = nn.Linear(dim, vocab_size, bias=False)

for layer_idx in range(n_layers):
    layer = nn.Module()
    layer.attention = Attention(dim, n_heads)
    layer.attention_norm = RMSNorm(dim, float(norm_eps))
    layer.ffn = FeedForward(dim, hidden_dim)
    layer.ffn_norm = RMSNorm(dim, float(norm_eps))
    setattr(model, f'layers.{layer_idx}', layer)

# Copy weights
model.tok_embeddings.weight = nn.Parameter(model_state['tok_embeddings.weight'].float())
model.norm.weight = nn.Parameter(model_state['norm.weight'].float())
model.output.weight = nn.Parameter(model_state['output.weight'].float())

for layer_idx in range(n_layers):
    layer = getattr(model, f'layers.{layer_idx}')
    layer.attention.wq.weight = nn.Parameter(model_state[f'layers.{layer_idx}.attention.wq.weight'].float())
    layer.attention.wk.weight = nn.Parameter(model_state[f'layers.{layer_idx}.attention.wk.weight'].float())
    layer.attention.wv.weight = nn.Parameter(model_state[f'layers.{layer_idx}.attention.wv.weight'].float())
    layer.attention.wo.weight = nn.Parameter(model_state[f'layers.{layer_idx}.attention.wo.weight'].float())
    layer.attention_norm.weight = nn.Parameter(model_state[f'layers.{layer_idx}.attention_norm.weight'].float())
    layer.ffn.w1.weight = nn.Parameter(model_state[f'layers.{layer_idx}.feed_forward.w1.weight'].float())
    layer.ffn.w2.weight = nn.Parameter(model_state[f'layers.{layer_idx}.feed_forward.w2.weight'].float())
    layer.ffn.w3.weight = nn.Parameter(model_state[f'layers.{layer_idx}.feed_forward.w3.weight'].float())
    layer.ffn_norm.weight = nn.Parameter(model_state[f'layers.{layer_idx}.ffn_norm.weight'].float())

model.cuda()
model.eval()

# Generate
import torch.nn.functional as F
torch.manual_seed(42)

input_ids = torch.tensor(tokens).unsqueeze(0).cuda()
with torch.no_grad():
    x = model.tok_embeddings(input_ids)
    for layer_idx in range(n_layers):
        layer = getattr(model, f'layers.{layer_idx}')
        h = layer.attention_norm(x)
        h = layer.attention(h, 0)
        x = x + h
        x = x + layer.ffn(layer.ffn_norm(x))
    x = model.norm(x)
    logits = model.output(x)
    probs = F.softmax(logits[:, -1] / 0.8, dim=-1)
    next_token = torch.multinomial(probs, 1).item()

# Map token to character
chars = [chr(i) for i in range(256)]
while len(chars) < vocab_size:
    chars.append(chr(255))

print(f'V2: {prompt}{chars[next_token]}')
" 2>&1
    
    echo ""
done
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "COMPARISON COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To run servers:"
echo "  bash start-v1-server.sh  (port 8084)"
echo "  bash start-v2-server.sh  (port 8085)"
echo ""
