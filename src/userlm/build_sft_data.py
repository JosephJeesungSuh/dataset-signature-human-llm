"""Build user-model SFT samples (App. D): the dialogue is flipped so that the model predicts user turns.

The intent is the system prompt; a conversation with n exchanges becomes n + 1 user utterances (the last
is <|endconversation|>) and n assistant responses, formatted with the base model's chat template. Labels
cover every user turn's content and its end-of-turn token; the intent, assistant turns, and turn headers
are masked with -100. Conversations must end with an assistant response.

    python -m src.userlm.prepare_tokenizer --base_model Qwen/Qwen2.5-7B-Instruct
    python -m src.userlm.build_sft_data --source wildchat_1m --base_model Qwen/Qwen2.5-7B-Instruct

Output: artifacts/userlm_data/<source>/sft/{split}_<model tag>_samples.jsonl (input_ids, labels) and an
aligned {split}_<model tag>_ids.txt with one conversation ID per sample, in corpus order.
"""
import argparse
import json
from multiprocessing import Pool

from tqdm import tqdm

from src.config import CORPUS, SOURCES, SPLITS, USER_BASE_MODELS, USERLM_DATA, model_tag
from src.userlm.chat_templates import CHAT_TEMPLATES, ENDCONV, TURN_MARKERS
from src.userlm.prepare_tokenizer import tokenizer_dir

_FLIPPER = None


class DialogueFlipper:
    def __init__(self, base_model):
        from transformers import AutoTokenizer
        self.base_model = base_model
        self.template = CHAT_TEMPLATES[base_model]
        self.user_header = TURN_MARKERS[base_model]["generation_prompt"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir(base_model)))
        assert ENDCONV in self.tokenizer.get_vocab()
        # Samples end with <|endoftext|> for Qwen models and with EOS otherwise.
        self.final_token = (self.tokenizer.convert_tokens_to_ids("<|endoftext|>") if base_model.startswith("Qwen/")
                            else self.tokenizer.eos_token_id)

    def flip(self, record):
        """{'input_ids', 'labels'} or None for ineligible conversations."""
        conv, intent = record.get("conversation") or [], (record.get("intent") or "").strip()
        if not conv or not intent or conv[-1]["role"] != "assistant":
            return None
        for i, t in enumerate(conv):
            if t["role"] != ("user" if i % 2 == 0 else "assistant") or not t["content"].strip():
                return None
        # Strip every message, as the template's own tokenization of "\n\n" boundaries depends on it.
        messages = ([{"role": "system", "content": intent}]
                    + [{"role": t["role"], "content": t["content"].strip()} for t in conv]
                    + [{"role": "user", "content": ENDCONV}])
        input_ids, labels = [], []
        for i, message in enumerate(messages):
            text = self.tokenizer.apply_chat_template(messages[:i + 1], tokenize=False, add_generation_prompt=False,
                                                      chat_template=self.template)
            if message["role"] != "user":   # the header of the following user turn is context, not a target
                text += self.user_header
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            new = tokens[len(input_ids):]
            assert tokens[:len(input_ids)] == input_ids, "chat template is not prefix-stable"
            labels.extend(new if message["role"] == "user" else [-100] * len(new))
            input_ids.extend(new)
        input_ids.append(self.final_token)
        labels.append(-100)
        return {"input_ids": input_ids, "labels": labels}


def _init(base_model):
    global _FLIPPER
    _FLIPPER = DialogueFlipper(base_model)


def _flip(record):
    return record["conv_id"], _FLIPPER.flip(record)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES, required=True)
    ap.add_argument("--base_model", choices=USER_BASE_MODELS, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = USERLM_DATA / args.source
    (root / "sft").mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in SPLITS:
        intents = {}
        for line in (root / "intents" / f"{split}.jsonl").open():
            r = json.loads(line)
            intents[r["conv_id"]] = r
        order = [json.loads(line)["conv_id"] for line in (CORPUS / args.source / f"{split}.jsonl").open()]
        records = [intents[c] for c in order if c in intents]
        prefix = root / "sft" / f"{split}_{model_tag(args.base_model)}"
        n = 0
        with Pool(args.workers, initializer=_init, initargs=(args.base_model,)) as pool, \
                open(f"{prefix}_samples.jsonl", "w") as fs, open(f"{prefix}_ids.txt", "w") as fi:
            for conv_id, sample in tqdm(pool.imap(_flip, records, chunksize=64), total=len(records),
                                        desc=f"{args.source}/{split}"):
                if sample is not None:
                    fs.write(json.dumps(sample) + "\n")
                    fi.write(conv_id + "\n")
                    n += 1
        stats[split] = {"conversations": len(order), "with_intent": len(records), "samples": n}
        print(split, stats[split], flush=True)
    (root / "sft" / f"stats_{model_tag(args.base_model)}.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
