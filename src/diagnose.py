# src/diagnose_unk.py
import json
from pathlib import Path
import sentencepiece as spm

PROJECT_ROOT = Path(__file__).parent.parent
TOKENIZER = PROJECT_ROOT / "tokenizer" / "act.model"

# 检查两个数据源
CC_PATH = PROJECT_ROOT / "data" / "cc2026_skeleton" / "cc2026_skeleton.jsonl"
ACL_PATH = PROJECT_ROOT / "data" / "processed" / "all_chunks.jsonl"

sp = spm.SentencePieceProcessor()
sp.load(str(TOKENIZER))
unk_id = sp.unk_id()
print(f"tokenizer vocab={sp.vocab_size()}, unk_id={unk_id}")

def check_unk(path, name):
    total = unk_lines = unk_count = 0
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            text = json.loads(line).get("text", "")
            ids = sp.encode(text, out_type=int)
            line_unk = ids.count(unk_id)
            unk_count += line_unk
            if line_unk > 0:
                unk_lines += 1
                if len(samples) < 5:
                    samples.append((text[:120], line_unk))
    
    print(f"\n{name}:")
    print(f"  总样本: {total}")
    print(f"  含unk样本: {unk_lines} ({unk_lines/total*100:.1f}%)")
    print(f"  总unk数: {unk_count}")
    if samples:
        print("  样例:")
        for s, c in samples:
            print(f"    [unk×{c}] {s}...")

check_unk(CC_PATH, "CC骨架")
check_unk(ACL_PATH, "ACL清洗")