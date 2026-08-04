"""
act2-moe (actually transformer 2号机 - mixture of experts)
最小定义版本：2 专家、Top-1 路由、FFN 宽度与 Dense 一致
"""

import torch
import torch.nn as nn

from model import multi_head_attention, feed_forward

default_config = {
    "vocab_size": 1024,
    "d_model": 64,
    "d_ff": 256,
    "num_layers": 1,
    "num_heads": 1,
    "num_experts": 2,
    "max_len": 512,
    "dropout": 0.1,
}


class moe_feed_forward(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            feed_forward(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

    def forward(self, x):
        route = self.router(x)
        route = torch.softmax(route, dim=-1)
        topk_val, topk_idx = torch.topk(route, 1, dim=-1)

        out = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (topk_idx.squeeze(-1) == i)
            if mask.any():
                tokens = x[mask]
                expert_out = self.experts[i](tokens)
                w = topk_val[mask]  # [N, 1]，无需再 unsqueeze
                out[mask] = w * expert_out
        return out


class act2_moe_layer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts=2, dropout=0.1):
        super().__init__()
        self.self_attn = multi_head_attention(d_model, num_heads, dropout)
        self.moe_ffn = moe_feed_forward(d_model, d_ff, num_experts, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, mask))
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.moe_ffn(x))
        return x


class act2_moe(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=1, d_ff=256,
                 num_layers=1, num_experts=2, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            act2_moe_layer(d_model, num_heads, d_ff, num_experts, dropout)
            for _ in range(num_layers)
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
    model = act2_moe(**default_config)
    n = sum(p.numel() for p in model.parameters())
    print(f"act2-moe params: {n:,}")
    dummy = torch.randint(0, default_config["vocab_size"], (2, 32))
    out = model(dummy)
    print(f"forward ok: {out.shape}")