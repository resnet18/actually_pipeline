"""
Common Crawl → 句子骨架提取器（warcio + fasttext + 去重 + 句首大写）
脚本位置: actually_pipeline/src/extract_sentence_skeleton.py
数据输出: actually_pipeline/data/cc2026_skeleton/cc2026_skeleton.jsonl

依赖: pip install warcio requests fasttext-wheel
注意: warcio 需要 numpy<2，若遇报错请 pip install "numpy<2"
"""
import random
import gzip
import re
import json
from pathlib import Path
from io import BytesIO

import requests
import fasttext
from warcio.archiveiterator import ArchiveIterator

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "cc2026_skeleton"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FTZ_PATH = PROJECT_ROOT / "lid.176.ftz"
# ==================================================

# ==================== 配置 ====================
CRAWL_ID = "CC-MAIN-2026-30"
NUM_WET_FILES = 10
MAX_DOCS_PER_FILE = 2000
LANG_THRESHOLD = 0.85

WET_PATHS_URL = f"https://data.commoncrawl.org/crawl-data/{CRAWL_ID}/wet.paths.gz"
FTZ_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
# ============================================


# ---------- fasttext 语言模型 ----------
def ensure_lang_model():
    if not FTZ_PATH.exists():
        print(f"[+] Downloading fasttext lid.176.ftz (1MB)...")
        r = requests.get(FTZ_URL)
        FTZ_PATH.write_bytes(r.content)
        print(f"    Saved to {FTZ_PATH}")
    return fasttext.load_model(str(FTZ_PATH))


LANG_MODEL = None


def is_english_fasttext(text: str) -> bool:
    if not text or len(text) < 30:
        return False
    sample = text.replace('\n', ' ')[:2000]
    labels, probs = LANG_MODEL.predict(sample, k=1)
    return labels[0] == '__label__en' and probs[0] >= LANG_THRESHOLD


# ---------- 核心过滤 ----------
def is_clean_sentence(sentence: str) -> bool:
    if re.search(r'[0-9\u0080-\uffff]', sentence):
        return False
    if re.search(r'[@#$%^&*()_+=\[\]{}|\\/<>~`":;]', sentence):
        return False

    words = sentence.split()
    if len(words) < 5 or len(words) > 15:
        return False

    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 2.0 or avg_len > 8.0:
        return False

    if re.search(r'[.!?,\'-]{2,}', sentence):
        return False

    if re.search(r'\b(ie|eg|etc|vs|viz|cf|et al)\b', sentence):
        return False

    if re.search(r'\b(www|http|com|org|net|html|pdf)\b', sentence):
        return False

    if not is_english_fasttext(sentence):
        return False

    return True


def extract_skeleton_paragraph(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)

    clean_sentences = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > 1:
            sent = sent[0].upper() + sent[1:].lower()
        else:
            sent = sent.upper()
        if is_clean_sentence(sent):
            clean_sentences.append(sent)

    if len(clean_sentences) < 2:
        return ""

    return " ".join(clean_sentences)


def heuristic_doc_filter(text: str) -> bool:
    if not text or len(text) < 300:
        return False

    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
    if digit_ratio > 0.05:
        return False

    upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
    if upper_ratio > 0.15:
        return False

    ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
    if ascii_ratio < 0.95:
        return False

    if not is_english_fasttext(text):
        return False

    return True


# ---------- 下载与解析（warcio 流式） ----------
def get_wet_urls() -> list[str]:
    print(f"[+] Fetching wet.paths.gz for {CRAWL_ID}...")
    r = requests.get(WET_PATHS_URL, timeout=60)
    r.raise_for_status()
    paths = gzip.decompress(r.content).decode("utf-8").strip().splitlines()
    sampled = random.sample(paths, min(NUM_WET_FILES, len(paths)))
    print(f"    Total: {len(paths)}, Sampled: {len(sampled)}")
    return sampled


def process_wet_file(wet_path: str, writer, seen_texts: set):
    url = f"https://data.commoncrawl.org/{wet_path}"
    print(f"[+] {url.split('/')[-1]} ...")

    kept_docs = 0
    try:
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            records = ArchiveIterator(resp.raw, arc2warc=True)

            for record in records:
                if record.rec_type != 'conversion':
                    continue

                content = record.content_stream().read()
                if not content:
                    continue

                try:
                    text = content.decode('utf-8', errors='ignore')
                except Exception:
                    continue

                if not heuristic_doc_filter(text):
                    continue

                skeleton = extract_skeleton_paragraph(text)
                if not skeleton:
                    continue

                if skeleton in seen_texts:
                    continue
                seen_texts.add(skeleton)

                item = {
                    "text": skeleton,
                    "source": "cc2026-skeleton",
                    "url": record.rec_headers.get_header('WARC-Target-URI') or "unknown",
                }
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept_docs += 1

                if kept_docs >= MAX_DOCS_PER_FILE:
                    break

        print(f"    Kept {kept_docs} unique skeleton docs")
    except Exception as e:
        print(f"    ERROR: {e}")


def main():
    global LANG_MODEL
    LANG_MODEL = ensure_lang_model()

    random.seed(42)
    wet_urls = get_wet_urls()

    out_path = OUTPUT_DIR / "cc2026_skeleton.jsonl"
    seen_texts = set()

    with out_path.open("w", encoding="utf-8") as f:
        for wet_path in wet_urls:
            process_wet_file(wet_path, f, seen_texts)

    total = sum(1 for _ in out_path.open("r", encoding="utf-8"))
    print(f"\n[Done] {out_path}")
    print(f"       Total unique skeleton docs: {total}")


if __name__ == "__main__":
    main()