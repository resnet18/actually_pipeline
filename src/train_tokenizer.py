"""
训练 SentencePiece BPE Tokenizer
输出：tokenizer/act.model + tokenizer/act.vocab
"""

import sentencepiece as spm
import os
import glob

# ==================== 路径配置（修改这里） ====================
# 获取本文件所在目录（即 src/）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))

# 输入：原始文本文件目录（每个 .txt 是一篇论文的清洗后文本）
# 现在可以先用 ACL 2025 语料占位，2026 到了替换这个路径
RAW_TEXT_DIR = os.path.join(PROJECT_ROOT, "data", "acl_2026_texts")
# 合并后的临时大文本（tokenizer 训练需要单文件输入）
MERGED_TEXT_PATH = os.path.join(PROJECT_ROOT, "data", "acl_raw_merged.txt")
# 输出：tokenizer 模型前缀
TOKENIZER_PREFIX = os.path.join(PROJECT_ROOT, "tokenizer", "act")
# ============================================================

# ========== Tokenizer 超参数 ==========
VOCAB_SIZE = 1024          # 总词表大小（包含 special tokens）
CHARACTER_COVERAGE = 0.9995
MODEL_TYPE = "bpe"         # 子词切分算法：bpe / unigram / word / char

# Special Tokens（预留，SFT 阶段直接启用）
# 注意：这些 token 的 piece 字符串里不能有空格
SPECIAL_TOKENS = ["<|<|endoftext|>", "|<|user|>", "|<|assistant|>"]
# =====================================


def merge_text_files(input_dir, output_path):
    """合并目录下所有 .txt 为一个文件"""
    files = glob.glob(os.path.join(input_dir, "*.txt"))
    if not files:
        raise FileNotFoundError(
            f"语料目录为空: {input_dir}\n"
            f"请先把清洗后的论文文本 (.txt) 放进这个目录，再运行本脚本。"
        )
    
    print(f"找到 {len(files)} 个文本文件，合并中...")
    with open(output_path, "w", encoding="utf-8") as out:
        for i, f in enumerate(files):
            with open(f, "r", encoding="utf-8") as inp:
                out.write(inp.read() + "\n")
            if (i + 1) % 100 == 0:
                print(f"  已处理 {i + 1}/{len(files)}")
    print(f"合并完成: {output_path}")


def train_tokenizer():
    # 1. 合并文本
    merge_text_files(RAW_TEXT_DIR, MERGED_TEXT_PATH)
    
    # 2. 确保输出目录存在
    os.makedirs(os.path.dirname(TOKENIZER_PREFIX), exist_ok=True)
    
    # 3. 训练
    print(f"\n开始训练 Tokenizer...")
    print(f"  词表大小: {VOCAB_SIZE}")
    print(f"  算法: {MODEL_TYPE}")
    print(f"  Special tokens: {SPECIAL_TOKENS}")
    
    spm.SentencePieceTrainer.train(
        input=MERGED_TEXT_PATH,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=CHARACTER_COVERAGE,
        model_type=MODEL_TYPE,
        user_defined_symbols=SPECIAL_TOKENS,
        pad_id=0,
        eos_id=1,              # <|endoftext|> 默认 id=1
        unk_id=2,
        # 其他 special tokens 的 id 会在 3 以后自动分配
    )
    
    print(f"\n训练完成！")
    print(f"  Model: {TOKENIZER_PREFIX}.model")
    print(f"  Vocab: {TOKENIZER_PREFIX}.vocab")
    print(f"\n验证加载...")
    
    # 4. 快速验证
    sp = spm.SentencePieceProcessor()
    sp.load(f"{TOKENIZER_PREFIX}.model")
    print(f"  实际词表大小: {sp.vocab_size()}")
    print(f"  '|<|endoftext|>' id: {sp.piece_to_id('|<|endoftext|>')}")
    print(f"  '|<|user|>' id: {sp.piece_to_id('|<|user|>')}")
    print(f"  '|<|assistant|>' id: {sp.piece_to_id('|<|assistant|>')}")
    
    # 编码测试
    test_text = "The attention mechanism is all you need."
    encoded = sp.encode(test_text, out_type=str)
    print(f"  编码测试: {encoded}")


if __name__ == "__main__":
    train_tokenizer()