import json
import re
from pathlib import Path

JSONL = Path(__file__).parent.parent / "data" / "processed" / "all_chunks.jsonl"

patterns = [
    (r'\([^()]*?\d{4}[^()]*?\)', "圆括号年份"),
    (r'\[[^\[\]]*?\d{4}[^\[\]]*?\]', "方括号年份"),
    (r'\bet\s+al', "et al"),
    (r'\b(?:Figure|Fig\.|Table|Tbl\.|Appendix|Algorithm|Alg\.|Equation|Eq\.)\s*\d', "图表标记"),
    (r'\b(?:pp\.|Sec\.|Section|Ch\.|Chapter)\s*\d', "章节页码"),
    (r'\b(?:19|20)\d{2}[a-z]?\b', "孤立年份"),
    (r'https?://', "URL"),
    (r'arXiv:', "arXiv"),
    (r'\d+\s*\.\s*\d+\s*in\s+the\s+full\s+model', "图表残留句"),  # 针对你样例里的 "5.14 in the full model"
]

hits = {name: 0 for _, name in patterns}
samples = []

with open(JSONL, "r", encoding="utf-8") as f:
    for line in f:
        text = json.loads(line).get("text", "")
        for pat, name in patterns:
            if re.search(pat, text, re.IGNORECASE):
                hits[name] += 1
                if len(samples) < 10:
                    samples.append((name, text[:160]))

print("残留统计:")
for name, c in hits.items():
    print(f"  {name:20s} ... {c} 条")

if any(hits.values()):
    print(f"\n总计残留: {sum(hits.values())} 条")
    print("\n样例:")
    for name, s in samples:
        print(f"  [{name}] {s}...")
else:
    print("\n干净，可以重训了。")