"""Encode conversations with a frozen language model and store pooled representations (App. B).

Decoders (Qwen2.5): text + EOS, up to 32,768 tokens; hidden states of four layers (config.QWEN_LAYERS),
each pooled as the mean over all tokens ("mean_L<l>") and as the final EOS state ("last_L<l>").
Encoders (config.ENCODERS): final-layer [CLS] state, or masked mean with L2 normalization (MiniLM),
truncated to each encoder's context length.

One output directory per source: artifacts/embeddings/<source>--<model>[--<variant>]/ with
{split}__{feature}.npy (float16), {split}__meta.jsonl, and spec.json.

    python -m src.classification.embed --model Qwen/Qwen2.5-7B --sources wildchat_1m lmsys
    python -m src.classification.embed --model Qwen/Qwen2.5-7B --sources wildchat_1m --view first_turn
    python -m src.classification.embed --model Qwen/Qwen2.5-7B --sources wildchat_1m \
        --data_root artifacts/data/corpus_original --variant original
    python -m src.classification.embed --model minilm --sources wildchat_1m lmsys sharechat
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np

from src.classification.data import VIEWS, load_split
from src.config import CORPUS, EMBED_MAX_LEN, EMBEDDINGS, ENCODERS, QWEN_LAYERS, SPLITS


def embedding_dir(source, model, variant=None):
    name = f"{source}--{model.rstrip('/').split('/')[-1]}"
    return EMBEDDINGS / (f"{name}--{variant}" if variant else name)


def _capture_layers(model, layers):
    """Forward hooks that keep only the requested hidden states. Index l follows HF's hidden_states:
    l = output of decoder block l, the last index = output after the final norm."""
    n = model.config.num_hidden_layers
    captured = {}

    def hook(layer):
        def fn(_module, _inputs, out):
            captured[layer] = out[0] if isinstance(out, tuple) else out
        return fn

    for layer in layers:
        if not 0 < layer <= n:
            raise ValueError(f"layer {layer} out of range 1..{n}")
        (model.norm if layer == n else model.layers[layer - 1]).register_forward_hook(hook(layer))
    return captured


def _batches(lengths, max_tokens, max_batch):
    """Length-sorted batches whose padded size (count x longest) stays within max_tokens."""
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) >= max_batch or (len(cur) + 1) * lengths[i] > max_tokens):
            batches.append(cur)
            cur = []
        cur.append(i)
    return batches + [cur] if cur else batches


def embed(args):
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer
    if args.model in ENCODERS:
        model_id, max_len, pool = ENCODERS[args.model]
        decoder, layers, pools = False, [None], [pool]
    else:
        model_id, max_len = args.model, args.max_len or EMBED_MAX_LEN
        decoder, layers, pools = True, args.layers or QWEN_LAYERS[model_id], ["mean", "last"]
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16).cuda().eval()
    captured = _capture_layers(model, layers) if decoder else None
    for source in args.sources:
        out = embedding_dir(source, model_id, args.variant)
        if (out / "spec.json").exists():
            print(f"skip {out} (exists)")
            continue
        out.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            t0 = time.time()
            texts, _, meta = load_split(split, [source], args.data_root, args.view, args.lowercase)
            ids = []
            for s in range(0, len(texts), 1024):
                if decoder:
                    batch = tok(texts[s:s + 1024], truncation=True, max_length=max_len - 1,
                                add_special_tokens=False)["input_ids"]
                    ids.extend(x + [tok.eos_token_id] for x in batch)
                else:
                    ids.extend(tok(texts[s:s + 1024], truncation=True, max_length=max_len)["input_ids"])
            feats = {}
            with torch.inference_mode():
                for idx in tqdm(_batches([len(x) for x in ids], args.max_tokens, args.batch_size),
                                desc=f"{out.name}/{split}", mininterval=10):
                    enc = tok.pad({"input_ids": [ids[i] for i in idx]}, padding=True, return_tensors="pt").to("cuda")
                    mask = enc["attention_mask"]
                    if decoder:
                        captured.clear()
                        model(**enc)
                        hidden = captured
                    else:
                        hidden = {None: model(**enc).last_hidden_state}
                    last = mask.sum(1) - 1
                    for layer in layers:
                        h = hidden[layer]
                        for pool in pools:
                            if pool == "mean":
                                m = mask.unsqueeze(-1).to(h.dtype)
                                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                            elif pool == "last":
                                v = h[torch.arange(len(idx), device=h.device), last]
                            elif pool == "mean_normalized":
                                v = (h.float() * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                                v = torch.nn.functional.normalize(v, p=2, dim=1)
                            else:   # cls
                                v = h[:, 0]
                            name = pool if layer is None else f"{pool}_L{layer}"
                            if name not in feats:
                                feats[name] = np.zeros((len(ids), v.shape[1]), dtype=np.float16)
                            feats[name][idx] = v.float().cpu().numpy().astype(np.float16)
                    del hidden, enc
            for name, arr in feats.items():
                assert np.isfinite(arr).all(), (out, split, name)
                np.save(out / f"{split}__{name}.npy", arr)
            with (out / f"{split}__meta.jsonl").open("w") as f:
                for m in meta:
                    f.write(json.dumps(m) + "\n")
            print(f"{out.name}/{split}: {len(ids)} conversations ({time.time() - t0:.0f}s)", flush=True)
        spec = {"source": source, "model": model_id, "view": args.view, "lowercase": args.lowercase,
                "data_root": str(Path(args.data_root).resolve()), "max_len": max_len, "append_eos": decoder,
                "layers": layers if decoder else None, "pooling": pools, "features": sorted(feats)}
        (out / "spec.json").write_text(json.dumps(spec, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B", help=f"HF decoder id or one of {sorted(ENCODERS)}")
    ap.add_argument("--sources", nargs="+", required=True, help="source directories under --data_root")
    ap.add_argument("--data_root", type=Path, default=CORPUS)
    ap.add_argument("--view", default="full_masked", choices=VIEWS)
    ap.add_argument("--lowercase", action="store_true", help="lowercase user messages")
    ap.add_argument("--variant", default=None, help="suffix of the output directory name (e.g. first_turn)")
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="override config.QWEN_LAYERS")
    ap.add_argument("--max_len", type=int, default=None, help="decoder context (default 32,768 tokens)")
    ap.add_argument("--max_tokens", type=int, default=65536, help="padded tokens per batch")
    ap.add_argument("--batch_size", type=int, default=32, help="max conversations per batch")
    embed(ap.parse_args())
