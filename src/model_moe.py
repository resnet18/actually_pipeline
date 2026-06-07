"""
AcT2-MoE (Actually Transformer - Mixture of Experts)
最小定义版本：2 专家、Top-1 路由、FFN 宽度与 Dense 一致（256）
"""

import torch
import torch.nn as nn
import math
import os

# ==================== 路径配置（修改这里） ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "act.model")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "model", "act_moe_base.pt")
# ============================================================

# ========== 超参数配置 ==========
VOCAB_SIZE = 1024
D_MODEL = 64
D_FF = 256          # 每个专家宽度 = Dense 的 FFN 宽度，不砍
NUM_LAYERS = 1
NUM_HEADS = 1
MAX_LEN = 512
DROPOUT = 0.1
NUM_EXPERTS = 2     # 最小 MoE：2 个专家
# ================================


# ========== 从 model.py 复用的组件（直接复制，避免 import 耦合） ==========
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.q_norm(Q)
        K = self.k_norm(K)
        Q, K = apply_rotary_pos_emb(Q, K, self.inv_freq, seq_len)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ========== MoE 核心：只替换 FFN ==========
class MoEFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        
        # Router：无 bias，最小参数
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # 专家池：每个专家 = 标准 FFN，宽度与 Dense 一致
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        
        # 1. 路由分数
        route = self.router(x)                 # (B, S, num_experts)
        route = torch.softmax(route, dim=-1)   # 归一化
        
        # 2. Top-1 选择（硬路由，最小定义版本）
        topk_val, topk_idx = torch.topk(route, 1, dim=-1)   # (B, S, 1)
        
        # 3. 初始化输出容器
        out = torch.zeros_like(x)
        
        # 4. 逐个专家处理（循环实现，可读性优先；极限尺度下效率无差异）
        for i in range(self.num_experts):
            mask = (topk_idx.squeeze(-1) == i)   # (B, S) bool
            if mask.any():
                tokens = x[mask]                   # (N, D) 被选中的 token
                expert_out = self.experts[i](tokens)  # (N, D)
                # 用路由权重缩放（保留梯度流）
                w = topk_val[mask].unsqueeze(-1)   # (N, 1)
                out[mask] = w * expert_out
        
        return out


class AcTMoELayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_experts=2, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.moe_ffn = MoEFeedForward(d_model, d_ff, num_experts, dropout)
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


class AcT2MoE(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=1, d_ff=256,
                 num_layers=1, num_experts=2, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            AcTMoELayer(d_model, num_heads, d_ff, num_experts, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        # Weight Tying（与 Dense 一致）
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
        seq_len = x.size(1)
        x = self.token_embedding(x)
        x = self.dropout(x)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_final(x)
        return self.lm_head(x)


if __name__ == "__main__":
    # 快速测试
    model = AcT2MoE(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, NUM_EXPERTS, MAX_LEN, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AcT-MoE 参数量: {n_params:,}")
    
    # 对比 Dense
    from model import AcT
    dense = AcT(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_LEN, DROPOUT)
    dense_params = sum(p.numel() for p in dense.parameters())
    print(f"AcT-Dense 参数量: {dense_params:,}")
    print(f"MoE 增量: {n_params - dense_params:,} (+{(n_params/dense_params - 1)*100:.1f}%)")
    
    # Forward 测试
    dummy = torch.randint(0, VOCAB_SIZE, (2, 32))
    out = model(dummy)
    print(f"输出形状: {out.shape} (预期: [2, 32, {VOCAB_SIZE}])")
    assert out.shape == (2, 32, VOCAB_SIZE)
    print("Forward 测试通过。")