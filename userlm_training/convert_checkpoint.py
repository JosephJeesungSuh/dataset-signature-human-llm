"""Convert the latest sharded FSDP checkpoint of a training run to a Hugging Face model.

The trainer saves <root>/fine-tuned-<model>-step-<N> during training and <root>/fine-tuned-<model> at the
end of an epoch; the most recently saved one is converted.

    python userlm_training/convert_checkpoint.py \
        --checkpoint_root artifacts/models/userlm_wildchat_1m_Qwen--Qwen2.5-7B-Instruct \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --tokenizer_path artifacts/models/tokenizers/Qwen--Qwen2.5-7B-Instruct \
        --output_path artifacts/models/userlm_wildchat_1m_Qwen--Qwen2.5-7B-Instruct_hf
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "llama-cookbook" / "src"))
from llama_cookbook.model_checkpointing.checkpoint_handler import load_sharded_model_single_gpu
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def latest_checkpoint(root):
    candidates = [p for p in Path(root).glob("fine-tuned-*") if list(p.glob("*.distcp"))]
    if not candidates:
        raise FileNotFoundError(f"no sharded checkpoint under {root}")
    return max(candidates, key=lambda p: max(f.stat().st_mtime for f in p.glob("*.distcp")))


def convert(checkpoint, model_name, tokenizer_path, output_path):
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name))
    # Training may have resized the embeddings to fit <|endconversation|>; match the checkpoint's shape.
    meta = FileSystemReader(str(checkpoint)).read_metadata().state_dict_metadata
    vocab = meta[next(k for k in meta if k.endswith("embed_tokens.weight"))].size[0]
    if vocab != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(vocab)
    model = load_sharded_model_single_gpu(model.to(torch.bfloat16), str(checkpoint))
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    AutoTokenizer.from_pretrained(tokenizer_path).save_pretrained(output_path)
    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size = vocab
    config.save_pretrained(output_path)
    model.save_pretrained(output_path)
    print(f"converted {checkpoint} -> {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_root", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--tokenizer_path", required=True)
    ap.add_argument("--output_path", required=True)
    a = ap.parse_args()
    convert(latest_checkpoint(a.checkpoint_root), a.model_name, a.tokenizer_path, a.output_path)
