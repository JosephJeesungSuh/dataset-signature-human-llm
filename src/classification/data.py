"""Classifier inputs and metrics.

A conversation is formatted as alternating "User: ..." / "Assistant: ..." lines (App. B). Views:
  full_masked    every assistant response replaced by <assistant_response> (default)
  full_unmasked  assistant responses visible (Table 3 ablation)
  first_turn     first user message only (Table 3 ablation)
  user_masked    every user message replaced by <user_query>, assistant text visible (App. E.3)
`lowercase=True` lowercases user messages (Table 3 ablation).
"""
import json
from pathlib import Path
from typing import Dict, List

from src.config import ASSISTANT_MASK, USER_MASK

VIEWS = ["full_masked", "full_unmasked", "first_turn", "user_masked"]


def serialize(conv: List[Dict], view: str = "full_masked", lowercase: bool = False) -> str:
    if view not in VIEWS:
        raise ValueError(view)
    lines = []
    for t in (conv[:1] if view == "first_turn" else conv):
        if t["role"] == "user":
            content = USER_MASK if view == "user_masked" else t["content"]
            lines.append(f"User: {content.lower() if lowercase and view != 'user_masked' else content}")
        else:
            content = t["content"] if view in ("full_unmasked", "user_masked") else ASSISTANT_MASK
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


def load_split(split: str, sources: List[str], root: Path, view: str = "full_masked", lowercase: bool = False):
    """Return (texts, labels, meta) for `split` of the given sources; label = position in `sources`."""
    texts, labels, meta = [], [], []
    for label, src in enumerate(sources):
        with (Path(root) / src / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                conv = r["conversation"]
                texts.append(serialize(conv, view, lowercase))
                labels.append(label)
                extra = r.get("meta") or {}
                meta.append({"source": src, "conv_id": r["conv_id"], "n_turns": len(conv),
                             "n_user_turns": sum(t["role"] == "user" for t in conv),
                             "n_words_user": r["n_words_user"],
                             "seed_conv_id": extra.get("seed_conv_id"), "intent_source": extra.get("intent_source")})
    return texts, labels, meta


def class_weights(labels, n_classes: int):
    """Inverse class-frequency weights, normalized to mean one."""
    import numpy as np
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return (w / w.mean()).tolist()


def metrics_from_preds(y_true, y_pred, names: List[str]) -> Dict:
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "per_class_recall": {n: float(cm[i, i] / max(1, cm[i].sum())) for i, n in enumerate(names)},
            "confusion_matrix": cm.tolist(), "class_names": list(names), "n": int(len(y_true)),
            "chance": 1.0 / len(names)}
