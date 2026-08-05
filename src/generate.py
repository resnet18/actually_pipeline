"""
act2 生成测试
用法: python src/generate.py
"""
from pathlib import Path

import torch
import sentencepiece as spm

from model import act2, default_config

# ==================== 配置（直接改这里） ====================
CHECKPOINT = Path(__file__).parent.parent / "model" / "act2-cpt-epoch2.pt"
TOKENIZER = Path(__file__).parent.parent / "tokenizer" / "act.model"

MAX_NEW = 64
TEMPERATURE = 0.7
TOP_K = 20
# ===========================================================


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", default_config)
    model = act2(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def generate(model, tokenizer, prompt, max_new=64, temperature=0.7, top_k=20):
    ids = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)

    # 获取特殊 token id（保险起见）
    pad_id = tokenizer.pad_id() if tokenizer.pad_id() > 0 else 0
    unk_id = tokenizer.unk_id() if tokenizer.unk_id() > 0 else 2
    eos_id = tokenizer.eos_id()

    with torch.no_grad():
        for _ in range(max_new):
            logits = model(input_ids)[:, -1, :] / temperature

            # 屏蔽特殊 token：禁止生成 pad 和 unk
            logits[:, pad_id] = float('-inf')
            logits[:, unk_id] = float('-inf')

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            if next_id.item() == eos_id:
                break

            input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not TOKENIZER.exists():
        raise FileNotFoundError(f"tokenizer not found: {TOKENIZER}")
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER))

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT}\n"
                                f"请先训练模型或修改 CHECKPOINT 路径。")

    model, cfg = load_model(CHECKPOINT, device)
    n = sum(p.numel() for p in model.parameters())
    print(f"loaded: {CHECKPOINT.name}")
    print(f"model: act2 | params={n:,} | vocab={cfg['vocab_size']} | device={device}")
    print(f"config: max_new={MAX_NEW}, temp={TEMPERATURE}, top_k={TOP_K}")
    print("-" * 40)
    print("输入 prompt 开始生成，输入 q / quit / exit 退出\n")

    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not prompt:
            continue
        if prompt.lower() in ("q", "quit", "exit", "bye"):
            print("bye")
            break

        text = generate(model, sp, prompt, MAX_NEW, TEMPERATURE, TOP_K)
        print(text)
        print("-" * 40)


if __name__ == "__main__":
    main()