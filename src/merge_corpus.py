import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

cc_path = PROJECT_ROOT / "data" / "cc2026_skeleton" / "cc2026_skeleton.jsonl"
acl_path = PROJECT_ROOT / "data" / "processed" / "all_chunks.jsonl"
out_path = PROJECT_ROOT / "data" / "merged_corpus.jsonl"

with open(out_path, "w", encoding="utf-8") as fout:
    # CC 骨架（通用语料）
    with open(cc_path, "r", encoding="utf-8") as f:
        for line in f:
            fout.write(line)
    # ACL 清洗（领域语料）
    with open(acl_path, "r", encoding="utf-8") as f:
        for line in f:
            fout.write(line)

print(f"合并完成: {out_path}")
# 统计
total = sum(1 for _ in open(out_path, "r", encoding="utf-8"))
print(f"总样本: {total}")