import os
import re
import json
from pathlib import Path
from typing import List


class ACLTextCleaner:
    """ACL Anthology 提取文本的清洗流水线"""

    def __init__(self, min_chunk_chars: int = 100, max_chunk_chars: int = 800):
        self.min_chunk = min_chunk_chars
        self.max_chunk = max_chunk_chars

        self.header_patterns = [
            re.compile(r'^Findings of the Association for Computational Linguistics.*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^The \d+th Annual Meeting of the Association for\n?', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^Association for Computational Linguistics.*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^.*©\d{4}\s*Association for Computational Linguistics.*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\d+\s*$', re.MULTILINE),
            re.compile(r'^ISBN\s+[\d\-]+\s*$', re.MULTILINE | re.IGNORECASE),
        ]

        self.end_markers = re.compile(
            r'(?:^|\n)\s*(?:References?|Acknowledgements?|Ethical Considerations|Limitations|Bibliography)\s*(?:\n|$)',
            re.IGNORECASE
        )

        self.figure_line = re.compile(
            r'^\s*(?:Figure|Fig\.|Table|Tbl\.|Algorithm|Alg\.|Equation|Eq\.)\s*[A-Z\d]',
            re.IGNORECASE
        )

        self.section_line = re.compile(
            r'^\s*(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)?)\s*[A-Z][a-zA-Z\s\-]{2,60}\s*$',
            re.MULTILINE
        )

        self.html_tag = re.compile(r'<[^>]+>')
        self.footnote_line = re.compile(r'^\s*[∗†‡§¶#]\s*[A-Z]')

        self.multiple_punct = re.compile(r'([.,;:!?])\1+')
        self.empty_parens = re.compile(r'\(\s*\)|\[\s*\]')
        self.leading_punct = re.compile(r'^\s*[.,;:!?)\]]+\s*')

    def should_skip_file(self, filename: str) -> bool:
        fname = filename.lower()
        return fname.endswith('.0.txt') or 'frontmatter' in fname or 'front' in fname

    def is_toc_by_content(self, text: str) -> bool:
        sample = text[:3000]
        if sample.count('. . .') > 10:
            return True
        keywords = ['table of contents', 'general chair', 'program chairs', 'organizing committee']
        return sum(1 for kw in keywords if kw in sample.lower()) >= 2

    def strip_headers_footers(self, text: str) -> str:
        text = self.html_tag.sub('', text)
        for pat in self.header_patterns:
            text = pat.sub('', text)
        return text

    def extract_body(self, text: str) -> str:
        m = re.search(r'(?:^|\n)\s*Abstract\s*[:\n]', text, re.IGNORECASE)
        if m:
            text = text[m.start():]
        else:
            m = re.search(r'(?:^|\n)\s*(?:1?\s*\.?\s*)?Introduction\s*\n', text, re.IGNORECASE)
            if m:
                text = text[m.start():]
        end_m = self.end_markers.search(text)
        if end_m:
            text = text[:end_m.start()]
        return text.strip()

    def remove_author_lines(self, text: str) -> str:
        lines = []
        for line in text.split('\n'):
            s = line.strip()
            if not s:
                lines.append(line)
                continue

            if (len(s) > 120 and s.count(',') >= 5 and ' and ' in s.lower()
                and re.match(r'^[A-Z][a-zA-Z\.\-]+', s)
                and re.search(r'\d{4}\.', s)):
                continue

            if re.search(r'\{.*?\}@|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', s):
                continue
            if (s.count(',') >= 2 and 'and' in s.lower()
                and len(s) < 150 and '.' not in s):
                continue
            lines.append(line)
        return '\n'.join(lines)

    def fix_line_breaks(self, text: str) -> str:
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        lines = text.split('\n')
        merged = []
        buf = ""
        for line in lines:
            s = line.strip()
            if not s:
                if buf:
                    merged.append(buf)
                    buf = ""
                merged.append("")
                continue
            if len(s) < 60 and not re.search(r'[.!?]$', s):
                buf = (buf + " " + s) if buf else s
            else:
                if buf:
                    buf = (buf + " " + s) if buf else s
                    merged.append(buf)
                    buf = ""
                else:
                    merged.append(s)
        if buf:
            merged.append(buf)
        text = '\n'.join(merged)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _cleanup_punctuation(self, text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' ([.,;:!?)}\]])', r'\1', text)
        text = re.sub(r'([({\[])\s+', r'\1', text)
        text = self.multiple_punct.sub(r'\1', text)
        text = self.empty_parens.sub('', text)
        lines = [self.leading_punct.sub('', line) for line in text.split('\n')]
        text = '\n'.join(lines)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +\n', '\n', text)
        text = re.sub(r'\n +', '\n', text)
        return text.strip()

    def strip_academic_markup(self, text: str) -> str:
        lines = [l for l in text.split('\n') if not self.footnote_line.match(l)]
        text = '\n'.join(lines)

        lines = [l for l in text.split('\n') if not self.figure_line.match(l)]
        text = '\n'.join(lines)

        lines = [l for l in text.split('\n') if not self.section_line.match(l)]
        text = '\n'.join(lines)

        text = re.sub(r'https?://\S*', '', text)
        text = re.sub(r'www\.\S*', '', text)
        text = re.sub(
            r'\b[a-zA-Z0-9][a-zA-Z0-9\-]*\.(?:com|org|net|edu|gov|io|info|biz|co\.[a-z]{2})(?:/\S*)?',
            '',
            text
        )
        text = re.sub(r'https?://', '', text)
        text = re.sub(r'arXiv:[\d\.]+(?:v\d+)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'10\.\d{4,}/\S+', '', text)

        for _ in range(20):
            new = re.sub(r'\([^()]{0,300}\)', '', text)
            if new == text:
                break
            text = new

        for _ in range(20):
            new = re.sub(r'\[[^\[\]]{0,100}\]', '', text)
            if new == text:
                break
            text = new

        words = text.split()
        clean = []
        i = 0
        while i < len(words):
            is_et_al = False
            if i + 1 < len(words) and words[i].lower() == 'et':
                next_core = words[i+1].lower().rstrip(',.;:)]')
                if next_core == 'al' or next_core.startswith('al.'):
                    is_et_al = True

            if is_et_al:
                removed = 0
                while removed < 3 and clean:
                    if re.match(r'^[A-Z][a-zA-Z\-\.]+,?$', clean[-1]):
                        clean.pop()
                        removed += 1
                    else:
                        break
                i += 2
                continue
            clean.append(words[i])
            i += 1
        text = ' '.join(clean)

        text = re.sub(
            r'\b(?:in|see|as discussed in|we discuss in|discussed in|described in|shown in|illustrated in|presented in)\s+(?:Section|Sec\.|Appendix|App\.|Chapter|Ch\.)\s*[A-Z\d][\w\.\-]*',
            '',
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r'\b(?:Section|Sec\.|Appendix|App\.|Chapter|Ch\.)\s*[A-Z\d][\w\.\-]*',
            '',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(r'\b\d+\.\d+(?:\.\d+)*\b', '', text)

        text = re.sub(
            r'\b(?:Figure|Fig\.|Table|Tbl\.|Algorithm|Alg\.|Equation|Eq\.|Section|Sec\.|Chapter|Ch\.|pp\.|pages?)\s*[A-Z\d][\w\.\-]*',
            '',
            text,
            flags=re.IGNORECASE
        )

        for _ in range(5):
            new = re.sub(r'\b(?:19|20)\d{2}[a-z]?\b', '', text)
            if new == text:
                break
            text = new

        text = self._cleanup_punctuation(text)

        return text.strip()

    def sanitize_ascii(self, text: str) -> str:
        text = re.sub(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
            r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251'
            r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U00002600-\U000026FF]+',
            '', text
        )
        cleaned = []
        for ch in text:
            o = ord(ch)
            if 32 <= o <= 126 or o in (9, 10, 13):
                cleaned.append(ch)
        text = ''.join(cleaned)
        text = self._cleanup_punctuation(text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            return []
        chunks = []
        current = ""
        for para in paragraphs:
            if len(para) > self.max_chunk:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= self.max_chunk:
                        current = (current + " " + sent) if current else sent
                    else:
                        if len(current) >= self.min_chunk:
                            chunks.append(current.strip())
                        current = sent
            else:
                if len(current) + len(para) + 2 <= self.max_chunk:
                    current = (current + "\n\n" + para) if current else para
                else:
                    if len(current) >= self.min_chunk:
                        chunks.append(current.strip())
                    current = para
        if current and len(current) >= self.min_chunk:
            chunks.append(current.strip())
        return chunks

    def clean(self, text: str) -> str:
        text = self.strip_headers_footers(text)
        text = self.extract_body(text)
        text = self.remove_author_lines(text)
        text = self.fix_line_breaks(text)
        text = self.strip_academic_markup(text)
        text = self.sanitize_ascii(text)
        return text


def process_directory(raw_dir: str, out_dir: str):
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cleaner = ACLTextCleaner(min_chunk_chars=100, max_chunk_chars=800)

    all_chunks = []
    stats = {"total": 0, "skipped": 0, "processed": 0, "discarded": 0, "chunks": 0, "chars": 0}

    txt_files = sorted(raw_path.rglob("*.txt"))
    stats["total"] = len(txt_files)
    print(f"发现 {len(txt_files)} 个 txt 文件（含子目录），开始处理...")

    for txt_file in txt_files:
        fname = txt_file.name

        if cleaner.should_skip_file(fname):
            stats["skipped"] += 1
            print(f"[SKIP] {txt_file.relative_to(raw_path)} -> 目录/前言文件")
            continue

        text = txt_file.read_text(encoding='utf-8')

        if cleaner.is_toc_by_content(text):
            stats["skipped"] += 1
            print(f"[SKIP] {txt_file.relative_to(raw_path)} -> 内容判定为目录")
            continue

        cleaned = cleaner.clean(text)

        if len(cleaned) < 200:
            stats["discarded"] += 1
            print(f"[DISCARD] {txt_file.relative_to(raw_path)}: 清洗后仅 {len(cleaned)} 字符")
            continue

        chunks = cleaner.chunk_text(cleaned)
        if not chunks:
            stats["discarded"] += 1
            print(f"[DISCARD] {txt_file.relative_to(raw_path)}: 无有效 chunks")
            continue

        stats["processed"] += 1
        stats["chunks"] += len(chunks)
        stats["chars"] += sum(len(c) for c in chunks)

        rel_path = txt_file.relative_to(raw_path)
        rel_stem = rel_path.with_suffix('')

        clean_file = out_path / "cleaned" / rel_path
        clean_file.parent.mkdir(parents=True, exist_ok=True)
        clean_file.write_text(cleaned, encoding='utf-8')

        chunk_name = str(rel_stem).replace('\\', '_').replace('/', '_') + ".jsonl"
        chunk_file = out_path / "chunks" / chunk_name
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_file, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(json.dumps({
                    "source": str(rel_path),
                    "chunk_id": i,
                    "text": chunk
                }, ensure_ascii=False) + '\n')
                all_chunks.append(chunk)

        print(f"[OK] {rel_path}: {len(chunks)} chunks, {len(cleaned)} chars")

    if all_chunks:
        with open(out_path / "all_chunks.jsonl", 'w', encoding='utf-8') as f:
            for chunk in all_chunks:
                f.write(json.dumps({"text": chunk}, ensure_ascii=False) + '\n')

    print("\n" + "="*50)
    print(f"总文件: {stats['total']} | 跳过: {stats['skipped']} | 处理: {stats['processed']} | 丢弃: {stats['discarded']}")
    print(f"总 chunks: {stats['chunks']} | 总字符: {stats['chars']:,}")
    if stats['chunks']:
        print(f"平均 chunk: {stats['chars']//stats['chunks']} chars")
    print(f"输出: {out_path}")
    print("="*50)


if __name__ == "__main__":
    RAW_DIR = r"D:\Projects\actually_pipeline\data\acl_2026_texts"
    OUT_DIR = r"D:\Projects\actually_pipeline\data\processed"
    process_directory(RAW_DIR, OUT_DIR)