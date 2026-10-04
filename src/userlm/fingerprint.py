"""Dataset classification on synthetic conversations (Sec. 3.1 Table 5, App. E.2 Table 10, App. E.3 Table 11).

  embed       frozen Qwen2.5-7B representations of the synthetic sets of every combination in
              config.ASSISTANT_IDENTITY_COMBOS (assistant responses masked; with --user_masked also the
              assistant-text-only view used in App. E.3)
  pairs       for every user-model pair of config.SYNTHETIC_PAIRS and one assistant:
                synth->synth  probe trained and tested on the two synthetic sets
                real->synth   probe trained (and selected) on the two real datasets, tested on the synthetic sets
                real->real    the same real probe's accuracy on the real test splits
  assistants  App. E.3: same intents and user model, different assistants; three binary comparisons and the
              three-way comparison, on user turns only (all / single-turn / multi-turn conversations) and on
              assistant turns only; balanced accuracy averaged over config.ASSISTANT_IDENTITY_COMBOS.

    python -m src.userlm.fingerprint embed --assistant qwen3p5-9b
    python -m src.userlm.fingerprint pairs --assistant qwen3p5-9b
    python -m src.userlm.fingerprint assistants
"""
import argparse
import itertools
import json
from argparse import Namespace

import numpy as np

from src.classification.embed import embed, embedding_dir
from src.classification.probe import run_probe
from src.config import (ASSISTANT_IDENTITY_COMBOS, ASSISTANTS, DISPLAY, EMBED_MODEL, RESULTS, SYNTHETIC,
                        SYNTHETIC_PAIRS, synthetic_name)

OUT = RESULTS / "fingerprint"


def embed_sets(assistants, user_masked):
    """Every (intent source, user model) combination of App. E.3; it includes all users of Sec. 3.1."""
    names = [synthetic_name(intent, user, tag) for tag in assistants for intent, user in ASSISTANT_IDENTITY_COMBOS]
    for view, variant in [("full_masked", None)] + ([("user_masked", "user_masked")] if user_masked else []):
        todo = [n for n in names if not (embedding_dir(n, EMBED_MODEL, variant) / "spec.json").exists()]
        if todo:
            embed(Namespace(model=EMBED_MODEL, sources=todo, data_root=SYNTHETIC, view=view, lowercase=False,
                            variant=variant, layers=None, max_len=None, max_tokens=65536, batch_size=32))


def pairs(assistant):
    rows = []
    for a, b, intent in SYNTHETIC_PAIRS:
        synth = [embedding_dir(synthetic_name(intent, u, assistant), EMBED_MODEL) for u in (a, b)]
        real = [embedding_dir(u, EMBED_MODEL) for u in (a, b)]
        base = OUT / f"pairs_{assistant}" / f"{a}__{b}"
        s2s = run_probe(synth, base / "synth_to_synth", names=[a, b])
        r2s = run_probe(real, base / "real_to_synth", names=[a, b], eval_dirs=synth)
        rows.append({"pair": f"{DISPLAY[a]} / {DISPLAY[b]}", "intent_source": DISPLAY[intent],
                     "real_to_real": r2s["test"]["accuracy"], "real_to_synth": r2s["eval"]["test"]["accuracy"],
                     "synth_to_synth": s2s["test"]["accuracy"]})
    (OUT / f"pairs_{assistant}.json").write_text(json.dumps(rows, indent=2))
    print(f"{'Dataset pair':<26}{'Real->real':>11}  {'Intent source':<14}{'Real->synth.':>13}{'Synth.->synth.':>15}")
    for r in rows:
        print(f"{r['pair']:<26}{100 * r['real_to_real']:11.1f}  {r['intent_source']:<14}"
              f"{100 * r['real_to_synth']:13.1f}{100 * r['synth_to_synth']:15.1f}")


def assistants():
    tags = list(ASSISTANTS)
    comparisons = [list(c) for c in itertools.combinations(tags, 2)] + [tags]
    conditions = {"all": (None, None), "single_turn": (None, "single"), "multi_turn": (None, "multi"),
                  "assistant_only": ("user_masked", None)}
    summary = {}
    for comp in comparisons:
        key = " vs. ".join(comp)
        summary[key] = {}
        for cond, (variant, turns) in conditions.items():
            scores = []
            for intent, user in ASSISTANT_IDENTITY_COMBOS:
                dirs = [embedding_dir(synthetic_name(intent, user, t), EMBED_MODEL, variant) for t in comp]
                out = OUT / "assistants" / "__".join(comp) / cond / f"{intent}__{user}"
                best = run_probe(dirs, out, names=comp, turns=turns)
                scores.append(best["test"]["balanced_accuracy"])
            summary[key][cond] = {"mean": float(np.mean(scores)), "min": float(np.min(scores)),
                                  "max": float(np.max(scores)), "chance": 1 / len(comp)}
            print(f"{key:<55} {cond:<15} {100 * np.mean(scores):5.1f} "
                  f"({100 * np.min(scores):.1f}-{100 * np.max(scores):.1f})", flush=True)
    (OUT / "assistants.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["embed", "pairs", "assistants"])
    ap.add_argument("--assistant", choices=list(ASSISTANTS), nargs="+", default=list(ASSISTANTS))
    ap.add_argument("--user_masked", action="store_true", help="embed: also the assistant-text-only view")
    a = ap.parse_args()
    if a.command == "embed":
        embed_sets(a.assistant, a.user_masked)
    elif a.command == "pairs":
        for tag in a.assistant:
            pairs(tag)
    else:
        assistants()
