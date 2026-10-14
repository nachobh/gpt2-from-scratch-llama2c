#!/usr/bin/env python3
"""Direct comparison of V1 and V2 models - with token mapping"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def get_vocabulary():
    """Vocabulary from the training data (sorted characters)"""
    with open('data/euromillon_merged.txt', 'r') as f:
        content = f.read()
    chars = sorted(set(content))
    return chars

def char_to_index(char, vocab):
    """Convert character to vocabulary index"""
    if char in vocab:
        return vocab.index(char)
    return 0  # Unknown character -> index 0

def index_to_char(index, vocab):
    """Convert vocabulary index back to character"""
    if 0 <= index < len(vocab):
        return vocab[index]
    return '?'  # Unknown index

def load_model(model_file):
    """Load model from PT file"""
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

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Move to CPU for stability
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

def build_and_load_model(model_file):
    """Build model and load weights"""
    model_state, config, dataset, vocab_size, n_layers, n_heads, dim, hidden_dim, max_seq_len, norm_eps = load_model(model_file)
    vocab = get_vocabulary()
    
    device = 'cpu'
    
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
    
    model.to(device)
    model.eval()
    return model, vocab, device

def generate(model, prompt, vocab, vocab_size, temperature=0.8, device='cpu'):
    """Generate next token with proper token mapping"""
    # Convert prompt to vocabulary indices
    token_indices = [char_to_index(c, vocab) for c in prompt]
    
    input_ids = torch.tensor(token_indices).unsqueeze(0).to(device)
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
        next_token_idx = torch.multinomial(probs, 1).item()
    
    # Convert index back to character
    next_char = index_to_char(next_token_idx, vocab)
    return prompt + next_char

if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else "models/euromillon-v2-gpt2.pt"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "1, 2, 3, 4, 5,"
    
    print(f"Loading model: {model_file}")
    model, vocab, device = build_and_load_model(model_file)
    print(f"Vocabulary size: {len(vocab)} chars")
    print(f"Device: {device}")
    print()
    
    for i in range(5):
        result = generate(model, prompt, vocab, len(vocab), temperature=0.8, device=device)
        print(f"Run {i+1}: {result}")
    
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
