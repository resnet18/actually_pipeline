"""
AcT2 (ActuallyTransformer) - 2号机
最小定义版本：子词级、Weight Tying、RoPE、QK-Norm、Dropout
"""

import torch
import torch.nn as nn
import math
import os

# ==================== 路径配置（修改这里） ====================
# 获取本文件所在目录（即 src/）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（向上退一级）
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
# Tokenizer 路径
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "act.model")
# 模型保存路径
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "model", "act_base.pt")
# ============================================================

# ========== 超参数配置（集中在此） ==========
VOCAB_SIZE = 1024      # 与 tokenizer 词表大小严格一致
D_MODEL = 64
D_FF = 256
NUM_LAYERS = 1
NUM_HEADS = 1
MAX_LEN = 512
DROPOUT = 0.1
# ==========================================


# ========== 最小 RoPE 实现 ==========
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


# ========== AcT 核心组件 ==========
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

        # QK-Norm：小模型防注意力 logits 爆炸
        self.q_norm = nn.LayerNorm(d_model)
        self.k_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # RoPE 频率预计算
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # QK-Norm
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        # RoPE
        Q, K = apply_rotary_pos_emb(Q, K, self.inv_freq, seq_len)

        # 拆头
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


class AcTLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
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


class AcT2(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=1, d_ff=256,
                 num_layers=1, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            AcTLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        # Weight Tying：lm_head 与 token_embedding 共享权重
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # 初始化
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
    # 快速测试：验证模型能跑通 forward
    model = AcT2(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_LEN, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AcT 参数量: {n_params:,}")
    
    # 随机输入测试
    dummy = torch.randint(0, VOCAB_SIZE, (2, 32))
    out = model(dummy)
    print(f"输出形状: {out.shape}  (预期: [2, 32, {VOCAB_SIZE}])")
    assert out.shape == (2, 32, VOCAB_SIZE)
    print("Forward 测试通过。")