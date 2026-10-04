"""Stage 1 of curation: load one source, keep structurally valid English conversations.

Released language labels are used when available, fastText (lid.176) otherwise.

    python -m src.curation.load --source wildchat_1m
"""
import argparse
import json
import time
from collections import Counter

from src.config import INTERIM, SOURCES
from src.curation.common import LangID, conversation_metadata, user_turns, validate_conversation, write_jsonl
from src.curation.sources import ENGLISH_LABELS, HAS_LANG_LABEL, LOADERS


def load_source(source: str):
    t0 = time.time()
    stats = Counter()
    lid = None if source in HAS_LANG_LABEL else LangID()

    def records():
        for rec in LOADERS[source]():
            stats["raw"] += 1
            reason = validate_conversation(rec["conversation"])
            if reason is not None:
                stats[f"invalid:{reason}"] += 1
                continue
            if lid is None:
                english = rec.get("lang_label") in ENGLISH_LABELS
            else:
                language, _ = lid.predict("\n".join(user_turns(rec["conversation"])))
                english = language == "en"   # top-1 language, no probability threshold
            if not english:
                stats["non_english"] += 1
                continue
            stats["english"] += 1
            rec.update(conversation_metadata(rec["conversation"]))
            rec["source"] = source
            yield rec

    write_jsonl(INTERIM / f"{source}.jsonl", records())
    summary = {"source": source, "stats": dict(stats), "seconds": round(time.time() - t0, 1)}
    (INTERIM / f"{source}.load_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES, required=True)
    load_source(ap.parse_args().source)
