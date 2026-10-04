"""Cross-dataset evaluation of user models (Sec. 3.3, Tables 7 and 12).

Scores the user turns of each dataset's test split (SFT samples built with the model's base tokenizer):
mean per-token negative log-likelihood over all user tokens (labels != -100) and its exponential, the
per-token perplexity. Conversations longer than 4,096 tokens are excluded, as in training.

    python -m src.userlm.eval_nll --model artifacts/models/userlm_wildchat_1m_Qwen--Qwen2.5-7B-Instruct_hf \
        --base_model Qwen/Qwen2.5-7B-Instruct --out artifacts/results/nll/qwen/wildchat_1m.json
"""
import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from src.config import SOURCES, USER_BASE_MODELS, USERLM_DATA, model_tag

MAX_LEN = 4096


def load_samples(path, max_len=MAX_LEN):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if len(d["input_ids"]) <= max_len and any(x != -100 for x in d["labels"][1:]):
                rows.append((d["input_ids"], d["labels"]))
    return rows


@torch.no_grad()
def nll(model, rows, pad_id, batch_tokens=16384, desc=""):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
    total, count, i = 0.0, 0, 0
    with tqdm(total=len(order), desc=desc, mininterval=10) as bar:
        while i < len(order):
            idx = [order[i]]
            i += 1
            while i < len(order) and (len(idx) + 1) * len(rows[order[i]][0]) <= batch_tokens:
                idx.append(order[i])
                i += 1
            width = max(len(rows[j][0]) for j in idx)
            inp = torch.full((len(idx), width), pad_id, dtype=torch.long)
            lab = torch.full((len(idx), width), -100, dtype=torch.long)
            att = torch.zeros((len(idx), width), dtype=torch.long)
            for r, j in enumerate(idx):
                ids, labels = rows[j]
                inp[r, :len(ids)], lab[r, :len(ids)], att[r, :len(ids)] = torch.tensor(ids), torch.tensor(labels), 1
            inp, lab, att = inp.cuda(), lab.cuda(), att.cuda()
            # Apply the LM head only where a user token is predicted, to avoid [B, L, V] logits.
            hidden = model.model(input_ids=inp, attention_mask=att).last_hidden_state
            target = lab[:, 1:]
            r_idx, p_idx = (target != -100).nonzero(as_tuple=True)
            h, tgt = hidden[r_idx, p_idx], target[r_idx, p_idx]
            for s in range(0, len(tgt), 2048):
                logits = model.lm_head(h[s:s + 2048]).float()
                total += float(torch.nn.functional.cross_entropy(logits, tgt[s:s + 2048], reduction="sum"))
            count += len(tgt)
            bar.update(len(idx))
    return {"nll_token_mean": total / count, "ppl_token": math.exp(total / count), "n_conv": len(rows),
            "n_user_tokens": count}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="converted user model directory")
    ap.add_argument("--base_model", choices=USER_BASE_MODELS, required=True)
    ap.add_argument("--test_sources", nargs="+", default=SOURCES)
    ap.add_argument("--test_root", type=Path, default=USERLM_DATA,
                    help="root with <source>/sft/test_<tag>_samples.jsonl")
    ap.add_argument("--batch_tokens", type=int, default=16384)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    results = json.loads(args.out.read_text()) if args.out.exists() else {"model": args.model, "per_test_set": {}}
    for source in args.test_sources:
        if source in results["per_test_set"]:
            continue
        rows = load_samples(args.test_root / source / "sft" / f"test_{model_tag(args.base_model)}_samples.jsonl")
        results["per_test_set"][source] = r = nll(model, rows, pad, args.batch_tokens, desc=source)
        print(f"{source}: nll {r['nll_token_mean']:.4f} ppl {r['ppl_token']:.2f} ({r['n_conv']} conversations)")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
