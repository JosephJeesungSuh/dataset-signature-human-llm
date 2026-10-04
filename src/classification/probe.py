"""Linear probe on frozen representations: the dataset-classifier recipe shared by all experiments (App. B).

For each stored representation (layer x pooling): standardize with training-split statistics, train a
multinomial logistic-regression head with class-weighted cross-entropy plus an L2 penalty on the weights,
using full-batch Adam (lr 0.05, 300 epochs, zero initialization), and keep the checkpoint with the best
validation balanced accuracy (checked every 10 epochs). The L2 strength in {1e-4, 1e-3, 1e-2} and the
representation are then chosen by validation balanced accuracy (equal to accuracy on the balanced
validation splits); the test split is never used for selection.

    # dataset classification (class order = order of the embedding directories)
    python -m src.classification.probe --classes wildchat_1m--Qwen2.5-7B lmsys--Qwen2.5-7B \
        --out artifacts/results/probes/wildchat_1m__lmsys
    # real-to-synthetic: fit on real data, also evaluate on synthetic test conversations
    python -m src.classification.probe --classes lmsys--Qwen2.5-7B sharegpt--Qwen2.5-7B \
        --eval_classes SYNTH_LMSYS--Qwen2.5-7B SYNTH_SHAREGPT--Qwen2.5-7B --out ...
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.classification.data import class_weights, metrics_from_preds
from src.config import EMBEDDINGS, SEED, SPLITS

WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2)


def fit_logreg(Xtr, ytr, Xva, yva, n_classes, weights, wd, epochs=300, lr=0.05, device="cuda"):
    """Returns (best validation balanced accuracy, W, b)."""
    import torch
    from sklearn.metrics import balanced_accuracy_score
    Xtr_t, ytr_t, Xva_t = (torch.as_tensor(a, device=device) for a in (Xtr, ytr, Xva))
    W = torch.zeros(Xtr.shape[1], n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    wt = torch.tensor(weights, device=device, dtype=torch.float32)
    best = (-1.0, None, None)
    for epoch in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr_t @ W + b, ytr_t, weight=wt) + wd * (W * W).sum()
        loss.backward()
        opt.step()
        if epoch % 10 == 9:
            with torch.no_grad():
                score = balanced_accuracy_score(yva, (Xva_t @ W + b).argmax(1).cpu().numpy())
            if score > best[0]:
                best = (score, W.detach().clone(), b.detach().clone())
    return best


def predict_logits(model, X, device="cuda"):
    import torch
    out = []
    with torch.no_grad():
        W, b = torch.as_tensor(model["weights"], device=device), torch.as_tensor(model["bias"], device=device)
        for s in range(0, len(X), 8192):
            x = torch.as_tensor((X[s:s + 8192] - model["mean"]) / model["std"], device=device)
            out.append((x @ W + b).cpu().numpy())
    return np.concatenate(out)


def fit_probe(X, y, n_classes, device="cuda"):
    """X, y: dicts with train/val arrays. Returns the selected head with its normalization."""
    mu = X["train"].mean(0, keepdims=True)
    sd = X["train"].std(0, keepdims=True) + 1e-6
    Xtr, Xva = (X["train"] - mu) / sd, (X["val"] - mu) / sd
    weights = class_weights(y["train"], n_classes)
    best = None
    for wd in WEIGHT_DECAYS:
        score, W, b = fit_logreg(Xtr, y["train"], Xva, y["val"], n_classes, weights, wd, device=device)
        if best is None or score > best[0]:
            best = (score, W, b, wd)
    _, W, b, wd = best
    return {"weights": W.cpu().numpy(), "bias": b.cpu().numpy(), "mean": mu, "std": sd, "weight_decay": wd}


def read_meta(directory, split):
    with (Path(directory) / f"{split}__meta.jsonl").open() as f:
        return [json.loads(line) for line in f]


def class_rows(dirs, split, fraction=None, turns=None):
    """Row indices used from each class directory."""
    rows = []
    for ci, d in enumerate(dirs):
        meta = read_meta(d, split)
        ix = np.arange(len(meta))
        if turns is not None:     # single-turn: one user message; multi-turn: two or more
            keep = [(m["n_user_turns"] == 1) == (turns == "single") for m in meta]
            ix = ix[np.asarray(keep, dtype=bool)]
        if fraction is not None and split == "train":   # nested seeded subsets per class
            ix = np.sort(np.random.default_rng(SEED + ci).permutation(ix)[:round(len(ix) * fraction)])
        rows.append(ix)
    return rows


def load_features(dirs, feature, split, rows=None):
    """Stack class directories; label = position in `dirs`."""
    xs, ys = [], []
    for ci, d in enumerate(dirs):
        arr = np.load(Path(d) / f"{split}__{feature}.npy", mmap_mode="r")
        ix = np.arange(len(arr)) if rows is None else rows[ci]
        xs.append(np.asarray(arr[ix], dtype=np.float32))
        ys.append(np.full(len(ix), ci, dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


def check_paired_intents(dirs):
    """Synthetic classes compared against each other must cover the same seed intents in every split."""
    seen = set()
    for split in SPLITS:
        keys = [{m["seed_conv_id"] for m in read_meta(d, split)} for d in dirs]
        if None in set.union(*keys):
            return
        if any(k != keys[0] for k in keys[1:]):
            raise ValueError(f"synthetic classes cover different intents in {split}; finish generation first")
        if seen & keys[0]:
            raise ValueError("an intent appears in more than one split")
        seen |= keys[0]


def run_probe(dirs, out, names=None, eval_dirs=None, fraction=None, turns=None, features=None, device=None,
              rows=None):
    """Fit and evaluate probes for every shared representation; return the validation-selected result.

    `rows` optionally fixes the rows used from each class: {split: [index array per class]}.
    """
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dirs = [Path(d) for d in dirs]
    names = names or [json.loads((d / "spec.json").read_text())["source"] for d in dirs]
    if turns is None and rows is None:
        check_paired_intents(dirs)
    shared = set.intersection(*[set(json.loads((d / "spec.json").read_text())["features"]) for d in dirs])
    features = sorted(shared if features is None else set(features) & shared)
    if not features:
        raise ValueError("no shared representations")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = rows or {split: class_rows(dirs, split, fraction, turns) for split in SPLITS}
    eval_rows = class_rows(eval_dirs, "test", turns=turns) if eval_dirs else None
    test_ids = np.concatenate([np.asarray([m["conv_id"] for m in read_meta(d, "test")])[ix]
                               for d, ix in zip(dirs, rows["test"])])
    best = None
    for feature in features:
        X, y = {}, {}
        for split in SPLITS:
            X[split], y[split] = load_features(dirs, feature, split, rows[split])
        model = fit_probe(X, y, len(dirs), device)
        logits = {s: predict_logits(model, X[s], device) for s in ("val", "test")}
        result = {"classes": names, "class_dirs": [d.name for d in dirs], "feature": feature,
                  "weight_decay": model["weight_decay"], "n_train": int(len(y["train"])), "fraction": fraction,
                  "turns": turns, **{s: metrics_from_preds(y[s], logits[s].argmax(1), names) for s in ("val", "test")}}
        if eval_dirs:
            Xe, ye = load_features(eval_dirs, feature, "test", eval_rows)
            result["eval"] = {"class_dirs": [Path(d).name for d in eval_dirs],
                              "test": metrics_from_preds(ye, predict_logits(model, Xe, device).argmax(1), names)}
        np.savez_compressed(out / f"{feature}__model.npz", **{k: v for k, v in model.items() if k != "weight_decay"})
        np.savez_compressed(out / f"{feature}__test_logits.npz", logits=logits["test"], labels=y["test"],
                            conv_ids=test_ids)
        (out / f"{feature}.json").write_text(json.dumps(result, indent=2))
        print(f"[{out.name}] {feature} wd={model['weight_decay']:g} val={result['val']['balanced_accuracy']:.4f} "
              f"test={result['test']['balanced_accuracy']:.4f}"
              + (f" eval={result['eval']['test']['balanced_accuracy']:.4f}" if eval_dirs else ""), flush=True)
        if best is None or result["val"]["balanced_accuracy"] > best["val"]["balanced_accuracy"]:
            best = result
        del X, y
    (out / "best.json").write_text(json.dumps(best, indent=2))
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", nargs="+", required=True, help="embedding directories (names under artifacts/embeddings)")
    ap.add_argument("--eval_classes", nargs="+", default=None,
                    help="extra test-only directories, one per training class in the same order")
    ap.add_argument("--names", nargs="+", default=None, help="class names (default: each directory's source)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fraction", type=float, default=None, help="use a nested fraction of each training class")
    ap.add_argument("--turns", choices=["single", "multi"], default=None,
                    help="restrict every split to single-turn or multi-turn conversations")
    ap.add_argument("--features", nargs="+", default=None)
    a = ap.parse_args()
    resolve = lambda names: [d if Path(d).is_dir() else EMBEDDINGS / d for d in names]
    if a.eval_classes and len(a.eval_classes) != len(a.classes):
        ap.error("--eval_classes needs one directory per training class")
    best = run_probe(resolve(a.classes), a.out, a.names, resolve(a.eval_classes) if a.eval_classes else None,
                     a.fraction, a.turns, a.features)
    print(f"selected {best['feature']}: test accuracy {best['test']['accuracy']:.4f}, "
          f"balanced {best['test']['balanced_accuracy']:.4f}")
