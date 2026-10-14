#!/usr/bin/env python3
"""Generate Euromillón combinations using V1 or V2 model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random

# Configuración de Euromillón
MAX_NUMBERS = 5  # Números principales (1-50)
MAX_STARS = 2    # Estrellas (1-12)
MIN_NUM = 1
MAX_NUM = 50
MIN_STAR = 1
MAX_STAR = 12
SEPARATOR = "|"

def get_vocabulary():
    """Vocabulary from the training data"""
    with open('data/euromillon_merged.txt', 'r') as f:
        content = f.read()
    chars = sorted(set(content))
    return chars

def char_to_index(char, vocab):
    if char in vocab:
        return vocab.index(char)
    return 0

def index_to_char(index, vocab):
    if 0 <= index < len(vocab):
        return vocab[index]
    return ''

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
    model_state = weights['model_state']
    config = weights['config']
    dataset = weights['dataset']
    
    vocab_size = dataset['vocab_size']
    n_layers = config['n_layers']
    n_heads = config['n_heads']
    dim = config['dim']
    hidden_dim = config['hidden_dim']
    max_seq_len = config['max_seq_len']
    norm_eps = config['norm_eps']
    
    return model_state, config, dataset, vocab_size, n_layers, n_heads, dim, hidden_dim, max_seq_len, norm_eps

def build_model(model_file):
    model_state, config, dataset, vocab_size, n_layers, n_heads, dim, hidden_dim, max_seq_len, norm_eps = load_model(model_file)
    vocab = get_vocabulary()
    
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
    
    return model, vocab

def is_valid_number(char, min_val, max_val, context):
    """Check if generated char could be a valid digit in context"""
    if char.isdigit():
        num = int(char)
        # If we already have 2 digits, this is invalid
        if context.get('num_digits', 0) >= 2:
            return False
        return min_val <= num <= max_val
    return False

def generate_combination(model, vocab, initial_prompt, max_len=30, temperature=0.8):
    """Generate a complete Euromillón combination"""
    result = initial_prompt
    context = {'phase': 'numbers', 'count': 0, 'num_digits': 0}
    
    for _ in range(max_len):
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
        
        # Update context
        if next_char == '|':
            context['phase'] = 'stars'
            context['count'] = 0
        elif next_char == ',':
            context['count'] += 1
            context['num_digits'] = 0
        elif next_char.isdigit():
            context['num_digits'] += 1
        
        # Stop conditions
        if context['phase'] == 'stars' and context['count'] >= 1:
            break
        if context['phase'] == 'numbers' and context['count'] >= 4:
            break
    
    return result.strip()

if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else "models/euromillon-v2-gpt2.pt"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "1, 2,"
    
    print(f"Model: {model_file}")
    print(f"Initial prompt: {prompt}")
    print(f"Generating 3 combinations...\n")
    
    model, vocab = build_model(model_file)
    
    for i in range(3):
        combo = generate_combination(model, vocab, prompt, max_len=30, temperature=0.8)
        print(f"  {i+1}: {combo}")
    
    del model
