"""Classifier-guided data selection (Sec. 3.4), adapting DSIR (Xie et al., 2023).

For a target dataset, the training splits of the other six datasets form the donor pool. A binary probe
(probe.py recipe on frozen Qwen2.5-7B representations, class-weighted) separates target from donor
conversations. With s_T(c) the predicted probability that donor conversation c belongs to the target, donors
are sampled without replacement with probability proportional to w_T(c) = s_T(c) / (1 - s_T(c)) (Gumbel top-k
on log w_T), at 50% and 100% of a 40K-conversation budget. The baseline samples donors uniformly at random.
Both conditions train on donor conversations only; validation and test files are the target's own.

    python -m src.datamix.select --target hh_rlhf

Outputs:
  artifacts/userlm_data/datamix/<target>__<dsir|random>_<pct>/sft/   SFT files for both base models
  artifacts/results/datamix/<target>/                                classifier, scores, and selections
"""
import argparse
import json
import random
import shutil

import numpy as np

from src.classification.data import metrics_from_preds
from src.classification.embed import embedding_dir
from src.classification.probe import fit_probe, load_features, predict_logits, read_meta
from src.config import EMBED_MODEL, RESULTS, SOURCES, USER_BASE_MODELS, USERLM_DATA, model_tag

BUDGET = 40000
PERCENTAGES = [50, 100]
SELECTION_SEED = 0


def sft_ids(source, base_model, split="train"):
    return (USERLM_DATA / source / "sft" / f"{split}_{model_tag(base_model)}_ids.txt").read_text().split()


def mix_dir(target, method, pct):
    return USERLM_DATA / "datamix" / f"{target}__{method}_{pct}" / "sft"


def training_rows(sources):
    """Embedding rows of the training conversations that have SFT samples (same set for both base models)."""
    rows = []
    for s in sources:
        ids = sft_ids(s, USER_BASE_MODELS[0])
        assert all(sft_ids(s, m) == ids for m in USER_BASE_MODELS[1:]), f"{s}: SFT samples differ across models"
        position = {m["conv_id"]: i for i, m in enumerate(read_meta(embedding_dir(s, EMBED_MODEL), "train"))}
        rows.append(np.asarray([position[c] for c in ids]))
    return rows


def train_classifier(target, out):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sources = [s for s in SOURCES if s != target] + [target]
    dirs = [embedding_dir(s, EMBED_MODEL) for s in sources]
    rows = {"train": training_rows(sources), "val": None, "test": None}
    features = sorted(set.intersection(*[set(json.loads((d / "spec.json").read_text())["features"]) for d in dirs]))
    best = None
    for feature in features:
        X, y = {}, {}
        for split in ("train", "val", "test"):
            X[split], labels = load_features(dirs, feature, split, rows[split])
            y[split] = (labels == len(sources) - 1).astype(np.int64)     # 1 = target
        model = fit_probe({"train": X["train"], "val": X["val"]}, y, 2, device)
        metrics = {s: metrics_from_preds(y[s], predict_logits(model, X[s], device).argmax(1), ["donor", target])
                   for s in ("val", "test")}
        print(f"[{target}] {feature} val balanced accuracy {metrics['val']['balanced_accuracy']:.4f}", flush=True)
        if best is None or metrics["val"]["balanced_accuracy"] > best[1]["val"]["balanced_accuracy"]:
            best = (feature, metrics, model)
        del X
    feature, metrics, model = best
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "classifier.npz", **{k: v for k, v in model.items() if k != "weight_decay"})
    (out / "classifier.json").write_text(json.dumps({"feature": feature, "weight_decay": model["weight_decay"],
                                                     **metrics}, indent=2))
    # Log-odds log(s_T / (1 - s_T)) of every conversation, per source and split.
    scores = {}
    for s, d in zip(sources, dirs):
        for split in ("train", "val", "test"):
            ix = rows["train"][sources.index(s)] if split == "train" else None
            x, _ = load_features([d], feature, split, None if ix is None else [ix])
            logits = predict_logits(model, x, device).astype(np.float64)
            scores[f"{s}__{split}"] = logits[:, 1] - logits[:, 0]
    np.savez_compressed(out / "scores.npz", **scores)


def select(target, out):
    donors = [s for s in SOURCES if s != target]
    with np.load(out / "scores.npz") as z:
        log_w = np.concatenate([z[f"{s}__train"] for s in donors])
    pool = [(s, i) for s in donors for i in range(len(sft_ids(s, USER_BASE_MODELS[0])))]
    assert len(pool) == len(log_w)
    gumbel = np.random.default_rng(SELECTION_SEED).gumbel(size=len(log_w))
    orders = {"dsir": np.argsort(-(log_w - log_w.max() + gumbel), kind="stable"),
              "random": np.random.default_rng(SELECTION_SEED).permutation(len(pool))}
    np.savez_compressed(out / "selection_orders.npz", **orders)
    selections = {}
    for method, order in orders.items():
        for pct in PERCENTAGES:
            chosen = [pool[i] for i in order[:BUDGET * pct // 100]]
            random.Random(SELECTION_SEED).shuffle(chosen)
            selections[method, pct] = chosen
            composition = {s: sum(c[0] == s for c in chosen) for s in donors}
            print(f"{target} {method} {pct}%: {len(chosen)} donor conversations {composition}", flush=True)
    for base in USER_BASE_MODELS:
        tag = model_tag(base)
        lines = {s: (USERLM_DATA / s / "sft" / f"train_{tag}_samples.jsonl").read_text().splitlines() for s in donors}
        ids = {s: sft_ids(s, base) for s in donors}
        for (method, pct), chosen in selections.items():
            dest = mix_dir(target, method, pct)
            dest.mkdir(parents=True, exist_ok=True)
            with open(dest / f"train_{tag}_samples.jsonl", "w") as fs, open(dest / f"train_{tag}_ids.txt", "w") as fi:
                for s, i in chosen:
                    fs.write(lines[s][i] + "\n")
                    fi.write(f"{s}/{ids[s][i]}\n")
            for split in ("val", "test"):
                for kind in ("samples.jsonl", "ids.txt"):
                    name = f"{split}_{tag}_{kind}"
                    shutil.copyfile(USERLM_DATA / target / "sft" / name, dest / name)
        del lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=SOURCES, required=True)
    a = ap.parse_args()
    out = RESULTS / "datamix" / a.target
    if not (out / "scores.npz").exists():
        train_classifier(a.target, out)
    select(a.target, out)
