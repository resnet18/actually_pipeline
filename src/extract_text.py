#!/usr/bin/env python3
"""
极简 PDF -> 文本提取器
for actually_pipeline 教程

用法：改下面 CONFIG，然后 python src/extract_text.py
"""

import json
import re
from pathlib import Path

import fitz
from tqdm import tqdm


# ==================== CONFIG ====================
# 和 download.py 的 cache 名保持一致
VOLUMES = [
    {"cache": "acl2026_long",     "text": "long"},
    {"cache": "acl2026_short",    "text": "short"},
    {"cache": "acl2026_findings", "text": "findings"},
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = PROJECT_ROOT / "data" / ".cache"
TEXT_ROOT = PROJECT_ROOT / "data" / "acl_2026_texts"
MIN_LENGTH = 200
# ================================================


def clean_text(text: str) -> str:
    ref_pat = re.compile(r'(?:^|\n)\s*(?:References?|REFERENCES?|Bibliography)\s*(?:\n|$)', re.I)
    m = ref_pat.search(text)
    if m:
        text = text[:m.start()]

    lines = text.split('\n')
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\d+$', line):
            continue
        if 'Proceedings of the' in line and 'Annual Meeting' in line:
            continue
        if line.startswith('https://doi.org/') or line.startswith('https://aclanthology.org/'):
            continue
        if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', line):
            continue
        out.append(line)

    text = '\n'.join(out)
    text = re.sub(r'([a-zA-Z,;])\n([a-z])', r'\1 \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_pdf(pdf_path: Path) -> str | None:
    try:
        doc = fitz.open(str(pdf_path))
        texts = [page.get_text() for page in doc if page.get_text()]
        doc.close()
        raw = '\n'.join(texts)
        return clean_text(raw) if raw.strip() else None
    except Exception:
        return None


def main():
    TEXT_ROOT.mkdir(parents=True, exist_ok=True)

    for cfg in VOLUMES:
        cache_dir = CACHE_ROOT / cfg["cache"]
        text_dir = TEXT_ROOT / cfg["text"]
        text_dir.mkdir(parents=True, exist_ok=True)

        meta = cache_dir / "metadata.json"
        if not meta.exists():
            print(f"[!] 跳过 {cfg['cache']}: 无 metadata")
            continue

        papers = json.loads(meta.read_text(encoding="utf-8"))
        print(f"[*] {cfg['cache']}: 提取 {len(papers)} 篇 -> {text_dir}")

        for p in tqdm(papers, desc=f"提取 {cfg['text']}", unit="篇"):
            pdf = cache_dir / (p["id"].replace("/", "_") + ".pdf")
            out = text_dir / (p["id"].replace("/", "_") + ".txt")

            if out.exists() and out.stat().st_size > 0:
                continue
            if not pdf.exists():
                continue

            text = extract_pdf(pdf)
            if text and len(text) >= MIN_LENGTH:
                out.write_text(text, encoding="utf-8")

    print("[✓] 提取完成")
    print(f"    文本目录: {TEXT_ROOT}")


if __name__ == "__main__":
    main()