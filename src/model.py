"""
act2 (actually transformer 2号机)
最小定义版本：子词级、Weight Tying、RoPE、QK-Norm、Dropout
"""

import torch
import torch.nn as nn
import math

default_config = {
    "vocab_size": 1024,
    "d_model": 64,
    "d_ff": 256,
    "num_layers": 1,
    "num_heads": 1,
    "max_len": 512,
    "dropout": 0.1,
}


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, inv_freq, seq_len):
    t = torch.arange(seq_len, device=q.device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class multi_head_attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, mask=None):
        b, s, _ = x.size()
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, self.inv_freq, s)

        q = q.view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.w_o(out)


class feed_forward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class act2_layer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = multi_head_attention(d_model, num_heads, dropout)
        self.feed_forward = feed_forward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, mask))
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        return x


class act2(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=1, d_ff=256,
                 num_layers=1, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            act2_layer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        s = x.size(1)
        x = self.token_embedding(x)
        x = self.dropout(x)

        mask = torch.tril(torch.ones(s, s, device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        return self.lm_head(x)


if __name__ == "__main__":
    model = act2(**default_config)
    n = sum(p.numel() for p in model.parameters())
    print(f"act2 params: {n:,}")
    dummy = torch.randint(0, default_config["vocab_size"], (2, 32))
    out = model(dummy)
    print(f"forward ok: {out.shape}")