"""Dataset classification experiments of Sec. 2.2: Table 2, Figure 2, Table 3, and Figure 3.

Jobs (probe recipe of probe.py on frozen representations; FOCAL = WC-1M + LMSYS + ShareChat):
  binary_*, ternary_*, incremental_*   Table 2 (incremental_7 gives the Figure 2 confusion matrix)
  focal_<view>                         Table 3 input ablations: assistant responses visible, markers restored,
                                       lowercased user messages, first user turn only
  structure_turns[_words]              Table 3: turn counts (+ total user word count) only
  size_<fraction>                      Figure 3 (left): nested fractions of each training split
  qwen_<size>, encoder_<name>          Figure 3 (right): other Qwen2.5 sizes and encoder families

    python -m src.classification.experiments list
    python -m src.classification.experiments embed --jobs 'binary_*' 'ternary_*' 'incremental_*'
    python -m src.classification.experiments run --jobs 'binary_*' 'ternary_*' 'incremental_*'
    python -m src.classification.experiments report
"""
import argparse
import fnmatch
import json
from argparse import Namespace

import numpy as np

from src.config import (BINARY_PAIRS, CORPUS, CORPUS_ORIGINAL, DISPLAY, EMBED_MODEL, ENCODERS, FOCAL,
                        INCREMENTAL, QWEN_LAYERS, RESULTS, SPLITS, TERNARY)

OUT = RESULTS / "classification"
# Table 3 views: job suffix -> embedding arguments.
FOCAL_VIEWS = {
    "full_unmasked": dict(view="full_unmasked"),
    "original": dict(data_root=CORPUS_ORIGINAL),
    "lowercase": dict(lowercase=True),
    "first_turn": dict(view="first_turn"),
}
FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50]


def jobs():
    out = {}
    add = lambda name, sources, model=EMBED_MODEL, variant=None, **kw: out.__setitem__(
        name, dict(sources=list(sources), model=model, variant=variant, **kw))
    for pair in BINARY_PAIRS:
        add("binary_" + "__".join(pair), pair)
    for triple in TERNARY:
        add("ternary_" + "__".join(triple), triple)
    for sources in INCREMENTAL:
        add(f"incremental_{len(sources)}", sources)
    for variant, kwargs in FOCAL_VIEWS.items():
        add(f"focal_{variant}", FOCAL, variant=variant, embed_kwargs=kwargs)
    out["structure_turns"] = dict(sources=FOCAL, structure=False)
    out["structure_turns_words"] = dict(sources=FOCAL, structure=True)
    for fraction in FRACTIONS:
        add(f"size_{fraction:g}", FOCAL, fraction=fraction)
    for model in QWEN_LAYERS:
        if model != EMBED_MODEL:
            add("qwen_" + model.split("-")[-1], FOCAL, model=model)
    for name in ENCODERS:
        add(f"encoder_{name}", FOCAL, model=name)
    return out


def select(patterns):
    chosen = {k: v for k, v in jobs().items() if any(fnmatch.fnmatchcase(k, p) for p in patterns)}
    if not chosen:
        raise SystemExit(f"no job matches {patterns}")
    return chosen


def model_id(job):
    return ENCODERS[job["model"]][0] if job["model"] in ENCODERS else job["model"]


def embed_jobs(patterns):
    from src.classification.embed import embed, embedding_dir
    todo = {}
    for job in select(patterns).values():
        if "structure" in job:
            continue
        key = (job["model"], job.get("variant"), json.dumps(job.get("embed_kwargs", {}), default=str))
        todo.setdefault(key, (job, set()))[1].update(
            s for s in job["sources"] if not (embedding_dir(s, model_id(job), job.get("variant")) / "spec.json").exists())
    for (model, variant, _), (job, sources) in todo.items():
        if sources:
            kw = {"data_root": CORPUS, "view": "full_masked", "lowercase": False, **job.get("embed_kwargs", {})}
            embed(Namespace(model=model, sources=sorted(sources), variant=variant, layers=None, max_len=None,
                            max_tokens=65536, batch_size=32, **kw))


def structure_job(name, job):
    """Logistic regression on turn counts (and total user word count) with the same probe recipe."""
    import torch
    from src.classification.data import metrics_from_preds
    from src.classification.probe import fit_probe, predict_logits
    from src.curation.common import read_jsonl
    X, y = {}, {}
    for split in SPLITS:
        rows, labels = [], []
        for ci, src in enumerate(job["sources"]):
            for r in read_jsonl(CORPUS / src / f"{split}.jsonl"):
                conv = r["conversation"]
                x = [sum(t["role"] == "user" for t in conv), sum(t["role"] == "assistant" for t in conv)]
                rows.append(x + [r["n_words_user"]] if job["structure"] else x)
                labels.append(ci)
        X[split], y[split] = np.asarray(rows, dtype=np.float32), np.asarray(labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = fit_probe(X, y, len(job["sources"]), device)
    names = job["sources"]
    result = {"classes": names, "feature": "counts", "weight_decay": model["weight_decay"],
              **{s: metrics_from_preds(y[s], predict_logits(model, X[s], device).argmax(1), names)
                 for s in ("val", "test")}}
    (OUT / name).mkdir(parents=True, exist_ok=True)
    (OUT / name / "best.json").write_text(json.dumps(result, indent=2))
    print(f"[{name}] test accuracy {result['test']['accuracy']:.4f}")


def run_jobs(patterns):
    from src.classification.embed import embedding_dir
    from src.classification.probe import run_probe
    for name, job in select(patterns).items():
        if (OUT / name / "best.json").exists():
            print(f"skip {name} (done)")
            continue
        if "structure" in job:
            structure_job(name, job)
            continue
        dirs = [embedding_dir(s, model_id(job), job.get("variant")) for s in job["sources"]]
        missing = [d.name for d in dirs if not (d / "spec.json").exists()]
        if missing:
            raise SystemExit(f"{name}: missing embeddings {missing}; run the embed command first")
        run_probe(dirs, OUT / name, names=job["sources"], fraction=job.get("fraction"))


def report():
    rows = []
    for name, job in jobs().items():
        path = OUT / name / "best.json"
        if path.exists():
            r = json.loads(path.read_text())
            rows.append((name, " + ".join(DISPLAY[s] for s in job["sources"]), r["feature"],
                         100 * r["test"]["chance"], 100 * r["test"]["accuracy"]))
    print(f"{'job':<46} {'datasets':<58} {'feature':<16} {'chance':>6} {'acc':>6}")
    for row in rows:
        print(f"{row[0]:<46} {row[1]:<58} {row[2]:<16} {row[3]:6.1f} {row[4]:6.1f}")
    path = OUT / "incremental_7" / "best.json"
    if path.exists():
        r = json.loads(path.read_text())["test"]
        cm = np.asarray(r["confusion_matrix"], dtype=float)
        print("\nSeven-way confusion matrix (rows: true source, % of row):")
        print(" " * 10 + "".join(f"{DISPLAY[c]:>10}" for c in r["class_names"]))
        for c, row in zip(r["class_names"], 100 * cm / cm.sum(1, keepdims=True)):
            print(f"{DISPLAY[c]:<10}" + "".join(f"{v:10.1f}" for v in row))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "embed", "run", "report"])
    ap.add_argument("--jobs", nargs="+", default=["*"], help="job names or quoted glob patterns")
    a = ap.parse_args()
    if a.command == "list":
        for name, job in jobs().items():
            print(f"{name:<46} {' + '.join(DISPLAY[s] for s in job['sources'])}")
    elif a.command == "embed":
        embed_jobs(a.jobs)
    elif a.command == "run":
        run_jobs(a.jobs)
    else:
        report()
