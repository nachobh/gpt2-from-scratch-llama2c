#!/usr/bin/env python3
"""Generate Euromillón combinations - improved version"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import re
import random

MAX_NUMBERS = 5
MAX_STARS = 2
MIN_NUM = 1
MAX_NUM = 50
MIN_STAR = 1
MAX_STAR = 12

def get_vocabulary():
    with open('data/euromillon_merged.txt', 'r') as f:
        content = f.read()
    return sorted(set(content))

def char_to_index(char, vocab):
    return vocab.index(char) if char in vocab else 0

def index_to_char(index, vocab):
    return vocab[index] if 0 <= index < len(vocab) else ''

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x_cpu = x.to('cpu')
        mean = x_cpu.mean(-1, keepdim=True)
        var = x_cpu.var(-1, keepdim=True, unbiased=False)
        x_cpu = (x_cpu - mean) / torch.sqrt(var + self.eps)
        return x_cpu * self.weight.to(x_cpu.device)

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        bsz, seq_len, _ = x.shape
        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
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

def load_model(model_file):
    weights = torch.load(model_file, weights_only=False, map_location='cpu')
    ms, cfg, ds = weights['model_state'], weights['config'], weights['dataset']
    return ms, cfg, ds

def build_model(model_file):
    ms, cfg, ds = load_model(model_file)
    vocab = get_vocabulary()
    vocab_size = ds['vocab_size']
    n_layers = cfg['n_layers']
    n_heads = cfg['n_heads']
    dim = cfg['dim']
    hidden_dim = cfg['hidden_dim']
    norm_eps = cfg['norm_eps']
    
    model = nn.Module()
    model.tok_embeddings = nn.Embedding(vocab_size, dim)
    model.norm = LayerNorm(dim, float(norm_eps))
    model.output = nn.Linear(dim, vocab_size, bias=False)
    for layer_idx in range(n_layers):
        layer = nn.Module()
        layer.attention = Attention(dim, n_heads)
        layer.attention_norm = LayerNorm(dim, float(norm_eps))
        layer.ffn = FeedForward(dim, hidden_dim)
        layer.ffn_norm = LayerNorm(dim, float(norm_eps))
        setattr(model, f'layers.{layer_idx}', layer)
    model.tok_embeddings.weight = nn.Parameter(ms['tok_embeddings.weight'].float())
    model.norm.weight = nn.Parameter(ms['norm.weight'].float())
    model.output.weight = nn.Parameter(ms['output.weight'].float())
    for layer_idx in range(n_layers):
        layer = getattr(model, f'layers.{layer_idx}')
        layer.attention.wq.weight = nn.Parameter(ms[f'layers.{layer_idx}.attention.wq.weight'].float())
        layer.attention.wk.weight = nn.Parameter(ms[f'layers.{layer_idx}.attention.wk.weight'].float())
        layer.attention.wv.weight = nn.Parameter(ms[f'layers.{layer_idx}.attention.wv.weight'].float())
        layer.attention.wo.weight = nn.Parameter(ms[f'layers.{layer_idx}.attention.wo.weight'].float())
        layer.attention_norm.weight = nn.Parameter(ms[f'layers.{layer_idx}.attention_norm.weight'].float())
        layer.ffn.w1.weight = nn.Parameter(ms[f'layers.{layer_idx}.feed_forward.w1.weight'].float())
        layer.ffn.w2.weight = nn.Parameter(ms[f'layers.{layer_idx}.feed_forward.w2.weight'].float())
        layer.ffn.w3.weight = nn.Parameter(ms[f'layers.{layer_idx}.feed_forward.w3.weight'].float())
        layer.ffn_norm.weight = nn.Parameter(ms[f'layers.{layer_idx}.ffn_norm.weight'].float())
    return model, vocab

def extract_numbers(text):
    """Extract all numbers from generated text"""
    return re.findall(r'\d+', text)

def generate_with_retries(model, vocab, prompt, max_attempts=10, temperature=0.3):
    """Try multiple times to get a valid combination"""
    best_result = None
    for attempt in range(max_attempts):
        result = prompt
        # Generate up to 40 characters
        for step in range(40):
            token_indices = [char_to_index(c, vocab) for c in result]
            input_ids = torch.tensor(token_indices).unsqueeze(0)
            with torch.no_grad():
                x = model.tok_embeddings(input_ids)
                for layer_idx in range(8):
                    layer = getattr(model, f'layers.{layer_idx}')
                    h = layer.attention_norm(x)
                    h = layer.attention(h)
                    x = x + h
                    x = x + layer.ffn(layer.ffn_norm(x))
                x = model.norm(x)
                logits = model.output(x)
                probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
            next_char = index_to_char(next_idx, vocab)
            result += next_char
            # Stop if we have too many non-numeric chars
            nums = extract_numbers(result)
            if len(nums) >= 7:
                break
            # Stop on newline or multiple spaces
            if next_char in ['\n', '\r']:
                break
        
        # Extract numbers
        nums = extract_numbers(result)
        # Filter valid numbers
        valid_nums = [int(n) for n in nums if n.isdigit()]
        
        if len(valid_nums) >= 5:
            numbers = sorted(set(valid_nums[:5]))
            # Pad if needed
            while len(numbers) < 5:
                numbers.append(random.randint(1, 50))
            numbers = sorted(set(numbers))[:5]
            while len(numbers) < 5:
                numbers.append(random.randint(1, 50))
            stars = sorted(random.sample(range(1, 13), min(2, len(valid_nums)-5))) if len(valid_nums) > 5 else [random.randint(1, 12)]
            return f"Numbers: {', '.join(str(n) for n in numbers[:5])} | Stars: {', '.join(str(s) for s in stars)}"
    
    return f"(no valid numbers found, generated: {result.strip()})"

if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else "models/euromillon-v2-gpt2.pt"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "1, 2,"
    num_combos = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    print(f"Model: {model_file}")
    print(f"Prompt: {prompt}")
    print(f"Temperature: 0.3 (low for consistency)")
    print(f"Generating {num_combos} combinations...\n")
    
    model, vocab = build_model(model_file)
    
    for i in range(num_combos):
        combo = generate_with_retries(model, vocab, prompt, temperature=0.3)
        print(f"  {i+1}: {combo}")
    
    del model
