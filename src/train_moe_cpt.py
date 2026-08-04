"""
act2-moe 继续预训练脚本（ACL 领域语料）
用法: python src/train_moe_cpt.py
"""
import os
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

import sentencepiece as spm
from model_moe import act2_moe, default_config

# ==================== 硬编码配置 ====================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "act.model"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "all_chunks.jsonl"
CKPT_DIR = PROJECT_ROOT / "model"
CKPT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 2
MAX_LEN = 512
SAVE_EVERY = 500
# ====================================================


class token_dataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_id()

        all_ids = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line).get("text", "")
                if not text:
                    continue
                all_ids.extend(tokenizer.encode(text, out_type=int))
                all_ids.append(self.eos_id)

        self.samples = []
        stride = max_len
        for i in range(0, len(all_ids) - max_len - 1, stride):
            self.samples.append(all_ids[i : i + max_len + 1])

        print(f"dataset: {len(self.samples)} blocks (max_len={max_len})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def get_lr(step, warmup, total, max_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    print(f"tokenizer loaded: vocab={sp.vocab_size()}")

    ds = token_dataset(DATA_PATH, sp, max_len=MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=(device.type == 'cuda'))

    cfg = default_config.copy()
    cfg["max_len"] = MAX_LEN
    model = act2_moe(**cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: act2-moe-cpt | params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                    betas=(0.9, 0.95), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # MoE CPT 自动 resume
    cpt_ckpt = CKPT_DIR / "act2-moe-cpt-latest.pt"
    pretrain_ckpt = CKPT_DIR / "act2-moe-latest.pt"
    start_epoch, step = 0, 0
    best_loss = float('inf')

    if cpt_ckpt.exists():
        ckpt = torch.load(cpt_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        best_loss = ckpt.get("loss", float('inf'))
        print(f"[resume] MoE-CPT interrupted, loaded {cpt_ckpt} (step={step}, epoch={start_epoch})")
    elif pretrain_ckpt.exists():
        ckpt = torch.load(pretrain_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"[resume] MoE-Pretrain → MoE-CPT, loaded {pretrain_ckpt}")
        step = 0
        start_epoch = 0
    else:
        raise FileNotFoundError(f"MoE 预训练权重不存在: {pretrain_ckpt}\n请先跑 python src/train_moe.py")

    total_steps = len(loader) * EPOCHS
    model.train()

    try:
        for epoch in range(EPOCHS):
            pbar = tqdm(loader, desc=f"CPT epoch {epoch+1}/{EPOCHS}")
            for x, y in pbar:
                x, y = x.to(device), y.to(device)

                lr = get_lr(step, 50, total_steps, LR)
                for g in optimizer.param_groups:
                    g['lr'] = lr

                logits = model(x)
                loss = criterion(logits.view(-1, cfg["vocab_size"]), y.view(-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                step += 1
                if step % 10 == 0:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})

                if step % SAVE_EVERY == 0:
                    latest = CKPT_DIR / "act2-moe-cpt-latest.pt"
                    torch.save({
                        "step": step, "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(), "config": cfg,
                    }, latest)
                    print(f"\n[save] {latest}")
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best = CKPT_DIR / "act2-moe-cpt-best.pt"
                        torch.save({
                            "step": step, "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": best_loss, "config": cfg,
                        }, best)

            ep_path = CKPT_DIR / f"act2-moe-cpt-epoch{epoch+1}.pt"
            torch.save({
                "step": step, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            }, ep_path)
            print(f"[epoch save] {ep_path}")

    except KeyboardInterrupt:
        interrupt = CKPT_DIR / "act2-moe-cpt-interrupt.pt"
        torch.save({
            "step": step, "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
        }, interrupt)
        print(f"\n[interrupt save] {interrupt}")
        raise

    print("CPT training complete.")


if __name__ == "__main__":
    main()