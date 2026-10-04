"""Shared curation helpers: normalization, hashing, conversation checks, language ID, JSONL I/O."""
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.config import LID_MODEL

# Opening-prompt template filter: drop prompts with a 7-gram that occurs more than 100 times.
NGRAM_N = 7
NGRAM_THRESHOLD = 100
# Full user texts (and opening prompts) of at least this many words link records across sources/splits.
MIN_WORDS_FOR_SHARING = 5


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def extract_ngrams(text: str, n: int = NGRAM_N) -> List[str]:
    words = text.lower().split()
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def user_turns(conv: List[Dict]) -> List[str]:
    return [t["content"] for t in conv if t["role"] == "user"]


def first_user_turn(conv: List[Dict]) -> str:
    return next((t["content"] for t in conv if t["role"] == "user"), "")


def conversation_metadata(conv: List[Dict]) -> Dict:
    """User-text hashes (lowercased, whitespace-collapsed) and conversation sizes."""
    text = "\n".join(user_turns(conv))
    return {
        "first_turn_hash": md5(norm_text(first_user_turn(conv))),
        "user_text_hash": md5(norm_text(text)),
        "n_words_user": len(text.split()),
        "n_turns": len(conv),
    }


def validate_conversation(conv: List[Dict]) -> Optional[str]:
    """None if the conversation starts with the user, strictly alternates user/assistant,
    has non-empty turns and at least one assistant turn; otherwise a short reason."""
    if not conv or len(conv) < 2:
        return "too_short"
    if conv[0]["role"] != "user":
        return "not_user_first"
    for i, t in enumerate(conv):
        if t.get("role") != ("user" if i % 2 == 0 else "assistant"):
            return "not_alternating"
        c = t.get("content")
        if not isinstance(c, str) or not c.strip():
            return "empty_turn"
    return None


class LangID:
    """fastText lid.176 language identification."""

    def __init__(self, model_path: Path = LID_MODEL):
        import fasttext
        if not model_path.is_file():
            raise FileNotFoundError(f"Download lid.176.bin from fastText and place it at {model_path}")
        self.model = fasttext.load_model(str(model_path))

    def predict(self, text: str):
        text = (text or "").replace("\n", " ").strip()[:3000]
        if not text:
            return None, 0.0
        preds = self.model.f.predict(text + "\n", 1, 0.0, "strict")
        if not preds:
            return None, 0.0
        prob, label = preds[0]
        return label.replace("__label__", ""), float(prob)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: Path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
