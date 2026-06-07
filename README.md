# actually_pipeline

A project about training lightweight GPT-like models from scratch on laptop CPUs, with a miniature version of the whole industrial pipeline.

---

## What is inside

- **AcT (Actually Transformer) 2**: Subword-level, decoder-only, ~115K params.  
  - Dense base (`model.py`)  
  - MoE base (`model_moe.py`, 2 experts, Top-1 routing, ~148K params)  
- **Tokenizer**: SentencePiece BPE, 1024 vocab, special tokens reserved for SFT (`<|endoftext|>`, `<|user|>`, `<|assistant|>`).
- **Training scripts**: `train_tokenizer.py` ready; `train.py` / `train_moe.py` pending corpus.

## Status

| Component | State |
|-----------|-------|
| Tokenizer | Ready |
| AcT-Dense | Frozen, waiting for ACL 2026 corpus |
| AcT-MoE | Frozen, waiting for ACL 2026 corpus |
| SFT / Post-training | Not started |

## Relation to actuallyX

- **actuallyX**: Standalone minimal-definition model architectures (frozen artifacts).
- **actually_pipeline**: Where those architectures are trained, evaluated, and optionally fine-tuned. The data-dependent half.
