"""
训练 SentencePiece BPE Tokenizer
输入：data/cc2026_skeleton/cc2026_skeleton.jsonl
输出：tokenizer/act.model + tokenizer/act.vocab
"""

import os
import json
import sentencepiece as spm

# ==================== 路径配置 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

# 清洗脚本的最终输出（每行一个 {"text": "..."}）
RAW_JSONL = os.path.join(PROJECT_ROOT, "data", "merged_corpus.jsonl")
# 合并后的临时大文本（SPM 训练需要单文件输入）
MERGED_TEXT_PATH = os.path.join(PROJECT_ROOT, "data", "cc_raw_merged.txt")
# 输出前缀
TOKENIZER_PREFIX = os.path.join(PROJECT_ROOT, "tokenizer", "act")
# =================================================

# ========== Tokenizer 超参数 ==========
VOCAB_SIZE = 1024          # 总词表大小（已包含 special tokens）
CHARACTER_COVERAGE = 0.9995
MODEL_TYPE = "bpe"

# Special Tokens（control_symbols：不拆分，不占 BPE 合并，纯占位）
# pad/eos/unk 由 SPM 内置 id 控制，其余 SFT 用 token 在这里预留
CONTROL_SYMBOLS = [
    "<|endoftext|>",   # 会被手动映射为 eos，或 SFT 时当对话结束符
    "<|user|>",        # 对话角色标记
    "<|assistant|>",    # 对话角色标记
]
# =====================================


def merge_jsonl_to_text(jsonl_path, output_path):
    """从 all_chunks.jsonl 提取 text 字段，合并为纯文本"""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"找不到 {jsonl_path}\n"
            f"请先运行 clean_acl.py 生成清洗后的语料，再执行本脚本。"
        )

    print(f"读取 {jsonl_path} ...")
    line_count = 0
    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if text:
                fout.write(text + "\n")
                line_count += 1
                if line_count % 10000 == 0:
                    print(f"  已处理 {line_count} 条 chunks")

    print(f"合并完成: {output_path}（共 {line_count} 条）")


def train_tokenizer():
    # 1. 合并文本
    merge_jsonl_to_text(RAW_JSONL, MERGED_TEXT_PATH)

    # 2. 确保输出目录存在
    os.makedirs(os.path.dirname(TOKENIZER_PREFIX), exist_ok=True)

    # 3. 训练
    print(f"\n开始训练 Tokenizer...")
    print(f"  词表大小: {VOCAB_SIZE}")
    print(f"  算法: {MODEL_TYPE}")
    print(f"  Control symbols: {CONTROL_SYMBOLS}")

    spm.SentencePieceTrainer.train(
        input=MERGED_TEXT_PATH,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=CHARACTER_COVERAGE,
        model_type=MODEL_TYPE,
        control_symbols=CONTROL_SYMBOLS,
        # 内置 special token id 固定
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,            # decoder-only 不需要 BOS
        # 训练稳定性
        num_threads=4,
        shuffle_input_sentence=True,
        input_sentence_size=0,   # 全用
    )

    print(f"\n训练完成！")
    print(f"  Model: {TOKENIZER_PREFIX}.model")
    print(f"  Vocab: {TOKENIZER_PREFIX}.vocab")

    # 4. 验证加载
    print(f"\n验证加载...")
    sp = spm.SentencePieceProcessor()
    sp.load(f"{TOKENIZER_PREFIX}.model")

    print(f"  实际词表大小: {sp.vocab_size()}")
    print(f"  <pad>  id: {sp.piece_to_id('<pad>')}  (pad_id=0)")
    print(f"  </s>   id: {sp.piece_to_id('</s>')}   (eos_id=1)")
    print(f"  <unk>  id: {sp.piece_to_id('<unk>')}  (unk_id=2)")
    for sym in CONTROL_SYMBOLS:
        print(f"  {sym:20s} id: {sp.piece_to_id(sym)}")

    # 5. 编码测试
    test = "Attention Is All You Need"
    pieces = sp.encode(test, out_type=str)
    ids = sp.encode(test)
    print(f"\n编码测试: '{test}'")
    print(f"  pieces: {pieces}")
    print(f"  ids:    {ids}")
    print(f"  解码回: {sp.decode(ids)}")


if __name__ == "__main__":
    train_tokenizer()