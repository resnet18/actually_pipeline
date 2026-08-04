# actually-pipeline

Training a 114K-parameter GPT-like model from scratch on a laptop, with a miniature but *real* industrial pipeline.

> Not "toy for pedagogy". Not "downscaled LLM".  
> This is what happens when you ask "what is the *minimum* definition of pretraining / CPT / SFT / RLHF(DPO) / RAG / agentic at 114K params?"  
> Spoiler: the model is too small to be aligned, but just large enough to align *your expectations*.

---

## Design Philosophy

- **Vibe-coding reliability**: If you can't debug it at 3 AM, it doesn't go in. No Flash Attention, no distributed training, no CUDA wizardry. Just `torch.nn` and stubbornness.
- **Minimal definition**: At 114K params, SFT is "conditional continuation", DPO is "pick the lower loss", RAG is "prepend context", and agentic is "prefix trigger". We don't pretend the model understands; we exploit its pattern-completion.
- **Reverse scaling**: Large models hide data quality problems behind parameters. Small models *expose* them. If your tokenizer drops a space, a 7B model shrugs; a 114K model writes `thecatsatonthemat` until you fix it. The failure is the diagnostic.
- **Infrastructure on ultrabook**: No GPU required, no WSL2, no CUDA toolkit, no conda environment archaeology. Just Windows + Python + `pip install`. If it can't run on an 11th Gen Intel Core i7-1165G7 while you're also running WeChat and 40 Chrome tabs, it's not in the pipeline.

---

## Project Structure

```
actually-pipeline/
├── src/
│   ├── extract_sentence_skeleton.py   # CC WET → clean English sentences (fasttext + dedup)
│   ├── clean_acl.py                   # ACL Anthology → stripped academic text
│   ├── check_clean.py                 # Audit script: count leftover citations/years/URLs
│   ├── train_tokenizer.py             # SPM BPE (1024 vocab, special tokens for SFT)
│   ├── train.py                       # Pretrain: act2-dense on CC skeleton
│   ├── train_moe.py                   # Pretrain: act2-moe on CC skeleton
│   ├── train_cpt.py                   # Continue Pretrain: act2-dense on ACL corpus
│   └── train_moe_cpt.py               # Continue Pretrain: act2-moe on ACL corpus
├── scripts/
│   └── pretrain.bat                   # One-click: base → MoE → CPT → MoE-CPT overnight
├── tokenizer/                         # act.model / act.vocab (generated)
├── data/
│   ├── cc2026_skeleton/               # CC-MAIN-2026-30 extracted (generated)
│   └── processed/                     # ACL 2026 cleaned chunks (generated)
└── model/                             # Checkpoints (generated)
```

---

## Environment Notes

- `warcio` (WET parsing) is incompatible with NumPy 2.x. Use `pip install "numpy<2"` if you see `Unable to avoid copy` errors.
- `fasttext-wheel` is the Windows-friendly build; do not install the plain `fasttext` package unless you enjoy compiling C++ extensions.

---

## The Pipeline (4 Stages)

| Stage | Script | Data | LR | Epochs | Resumes From | Output |
|-------|--------|------|----|--------|--------------|--------|
| 1. Pretrain (Dense) | `train.py` | CC skeleton | 5e-4 | 3 | — | `act2-latest.pt` |
| 2. Pretrain (MoE) | `train_moe.py` | CC skeleton | 5e-4 | 3 | — | `act2-moe-latest.pt` |
| 3. CPT (Dense) | `train_cpt.py` | ACL 2026 | 1e-4 | 2 | `act2-latest.pt` | `act2-cpt-latest.pt` |
| 4. CPT (MoE) | `train_moe_cpt.py` | ACL 2026 | 1e-4 | 2 | `act2-moe-latest.pt` | `act2-moe-cpt-latest.pt` |

Run all 4 in one go:

```powershell
.\scripts\pretrain.bat
```

Then go to sleep. The laptop will not.

---

## Data Pipeline

### Common Crawl (General Pretraining)
- **Source**: `CC-MAIN-2026-30` WET files (extracted plaintext, no HTML parsing)
- **Language**: `fasttext` `lid.176` filtered to English only (`__label__en`, confidence ≥ 0.85)
- **Cleaning**: Sentence-level heuristic filter (length, word count, avg word length, line dup ratio)
- **Normalization**: Only sentence-initial capitalization kept; intra-sentence proper nouns lowercased to save vocab space
- **Deduplication**: Exact-match dedup across all sampled WET segments

### ACL Anthology (Continue Pretraining)
- **Source**: ACL 2026 long/short/findings papers
- **Stripping**: Headers, footers, copyright lines, author emails, references, figure/table/appendix lines, inline citations `(Author, 2024)`, arXiv IDs, URLs, DOIs
- **Zero-backtracking `et al.` removal**: Because regex backtracking on 114K-scale data is a great way to learn what "catastrophic backtracking" means literally
- **Chunking**: 100–800 char sliding chunks for training

---

## Current Status

| Component | State | Notes |
|-----------|-------|-------|
| CC Extractor | Ready | WET parser + fasttext EN filter + dedup |
| ACL Cleaner | Ready | V3.2, audit-passed (160 residual artifacts acceptable) |
| Tokenizer | Waiting | Needs retrain on CC corpus (switched to sentence-case) |
| act2-dense pretrain | Blocked | Waiting for tokenizer + CC data |
| act2-moe pretrain | Blocked | Waiting for tokenizer + CC data |
| CPT (dense / MoE) | Blocked | Waiting for pretrain checkpoints |
| SFT / DPO / RAG / Agentic | Planned | Special tokens already reserved in tokenizer |

---

## Relation to actuallyX

- **actuallytransformer** (act): The *architecture* repo. Frozen artifacts, minimal definitions, no data.
- **actually-pipeline**: The *training* repo. This is where act gets its weights, its corpus, and its scars. The data-dependent half.

---

## Quick Start

# Extract CC skeleton (takes a while, downloads ~5-10GB WET)
python src/extract_sentence_skeleton.py

# Train tokenizer
python src/train_tokenizer.py

# Pretraining (or double-click scripts/pretrain.bat)
python src/train.py
python src/train_moe.py
python src/train_cpt.py
python src/train_moe_cpt.py

---

## Hardware Requirements

- **Device**: Any laptop from the last 5 years. We train on a Huawei MateBook D14 2021 (11th Gen Intel Core i7-1165G7 @ 2.80GHz).
- **OS**: Windows 10/11 (no WSL required, but works if you have it)
- **RAM**: 8GB+ (16GB recommended if you want to keep Chrome open)
- **Disk**: ~10GB for CC WET download, ~100MB for cleaned corpus
- **Configuration required**: None. `pip install -r requirements.txt` and go.
- **Patience**: More than your laptop's thermal capacity.