"""Save the base tokenizer with the added <|endconversation|> special token.

    python -m src.userlm.prepare_tokenizer --base_model Qwen/Qwen2.5-7B-Instruct
"""
import argparse

from src.config import MODELS, USER_BASE_MODELS, model_tag
from src.userlm.chat_templates import ENDCONV


def tokenizer_dir(base_model):
    return MODELS / "tokenizers" / model_tag(base_model)


def prepare(base_model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model)
    if ENDCONV not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [ENDCONV]})
    out = tokenizer_dir(base_model)
    tok.save_pretrained(out)
    assert ENDCONV in AutoTokenizer.from_pretrained(out).get_vocab()
    print(f"saved {base_model} tokenizer ({len(tok)} tokens) to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_model", choices=USER_BASE_MODELS, required=True)
    prepare(ap.parse_args().base_model)
