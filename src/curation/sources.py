"""Per-source loaders for the seven conversation datasets (App. A).

Each loader yields records of the form

  {"conv_id": str,            # unique within the source
   "group_id": str,           # unit kept within one split (user id when released, else conv_id)
   "model": str | None, "timestamp": str | None,
   "lang_label": str | None,  # language label released with the data, if any
   "presplit": str | None,    # released train/test membership (HH-RLHF only)
   "conversation": [{"role": "user" | "assistant", "content": str}, ...],
   "meta": {...}}

Roles are normalized here; structural validation and language filtering happen in load.py.
"""
import csv
import gzip
import json
import re
from collections import Counter
from os.path import commonprefix
from pathlib import Path

# Dataset snapshots used in the paper.
REVISIONS = {
    "allenai/WildChat-1M": "7d6490e462285cf85d91eabea0f9a954fbddcd1f",
    "allenai/WildChat-4.8M": "c827c6df8fcf008219ffaffa4d1dd77491099367",
    "lmsys/lmsys-chat-1m": "200748d9d3cddcc9d782887541057aca0b18c5da",
    "lmarena-ai/arena-human-preference-140k": "6322995ab34d7c2693e3f47dd13fa5caa0789a74",
    "tucnguyen/ShareChat": "a57c741e002684939f577594fdae1559832a584d",
    "anon8231489123/ShareGPT_Vicuna_unfiltered": "192ab2185289094fc556ec8ce5ce1e8e587154ca",
    "Anthropic/hh-rlhf": "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa",
}
WILDCHAT_REPOS = {"wildchat_1m": "allenai/WildChat-1M", "wildchat_4p8m": "allenai/WildChat-4.8M"}
# WildChat-4.8M contains WildChat-1M: the later class keeps only conversations whose hash is
# absent from the earlier release's English pool.
WILDCHAT_SUPERSEDED = {"wildchat_4p8m": "wildchat_1m"}
SHARECHAT_PLATFORMS = ("chatgpt", "claude", "gemini", "grok", "perplexity")

# Sources with a released language label; the others use fastText language identification.
HAS_LANG_LABEL = {"wildchat_1m", "wildchat_4p8m", "lmsys", "arena_2025", "sharechat"}
ENGLISH_LABELS = {"English", "english", "en", "EN", "eng"}


def hf_file(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=REVISIONS[repo]))


def hf_snapshot(repo: str) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, repo_type="dataset", revision=REVISIONS[repo],
                                  allow_patterns=["README.md", "data/*.parquet"]))


def _turns(conv, user_roles=("user",), assistant_roles=("assistant",), role_key="role", content_key="content"):
    out = []
    for t in conv:
        r = t.get(role_key)
        role = "user" if r in user_roles else "assistant" if r in assistant_roles else f"unknown:{r}"
        out.append({"role": role, "content": t.get(content_key) or ""})
    return out


# --------------------------------------------------------------------------- #
# WildChat-1M / WildChat-4.8M
# --------------------------------------------------------------------------- #
def _load_wildchat(source):
    import pyarrow.parquet as pq
    repo = WILDCHAT_REPOS[source]
    columns = ["conversation_hash", "hashed_ip", "country", "model", "timestamp",
               "language", "conversation", "redacted", "toxic", "turn"]
    for path in sorted(hf_snapshot(repo).glob("data/*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                ip = row.get("hashed_ip")
                # Missing identifiers must not merge unrelated users into one group.
                group = f"{ip}_{row.get('country') or 'unknown'}" if ip else f"{source}:{row['conversation_hash']}"
                yield {"conv_id": row["conversation_hash"], "group_id": group, "model": row.get("model"),
                       "timestamp": str(row.get("timestamp")), "lang_label": row.get("language"), "presplit": None,
                       "conversation": _turns(row["conversation"]),
                       "meta": {"country": row.get("country"), "redacted": row.get("redacted"),
                                "toxic": row.get("toxic"), "turn": row.get("turn")}}


def load_wildchat_1m():
    yield from _load_wildchat("wildchat_1m")


def load_wildchat_4p8m():
    yield from _load_wildchat("wildchat_4p8m")


# --------------------------------------------------------------------------- #
# LMSYS-Chat-1M
# --------------------------------------------------------------------------- #
def load_lmsys():
    from datasets import load_dataset
    repo = "lmsys/lmsys-chat-1m"
    ds = load_dataset(repo, revision=REVISIONS[repo], split="train")
    ds = ds.select_columns(["conversation_id", "model", "conversation", "turn", "language", "redacted"])
    for batch in ds.iter(batch_size=5000):
        for i in range(len(batch["conversation_id"])):
            yield {"conv_id": batch["conversation_id"][i], "group_id": batch["conversation_id"][i],
                   "model": batch["model"][i], "timestamp": None, "lang_label": batch["language"][i],
                   "presplit": None, "conversation": _turns(batch["conversation"][i]),
                   "meta": {"redacted": batch["redacted"][i], "turn": batch["turn"][i]}}


# --------------------------------------------------------------------------- #
# ShareGPT (the two HTML-cleaned files of ShareGPT_Vicuna_unfiltered)
# --------------------------------------------------------------------------- #
def load_sharegpt():
    repo = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    for fn in ["HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json",
               "HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json"]:
        data = json.loads(hf_file(repo, fn).read_text(encoding="utf-8"))
        for d in data:
            conv = _turns(d.get("conversations", []), user_roles=("human", "user"),
                          assistant_roles=("gpt", "chatgpt", "bing", "bard", "assistant"),
                          role_key="from", content_key="value")
            yield {"conv_id": d["id"], "group_id": d["id"], "model": None, "timestamp": None,
                   "lang_label": None, "presplit": None, "conversation": conv,
                   "meta": {"file": fn.split("/")[-1]}}


# --------------------------------------------------------------------------- #
# HH-RLHF: the chosen transcript; released train/test membership is kept.
# --------------------------------------------------------------------------- #
_HH_SPLIT = re.compile(r"\n\n(Human|Assistant): ")


def _parse_hh(transcript: str):
    parts = _HH_SPLIT.split(transcript)   # ['', 'Human', text, 'Assistant', text, ...]
    return [{"role": "user" if parts[i] == "Human" else "assistant", "content": parts[i + 1].strip()}
            for i in range(1, len(parts) - 1, 2)]


def _parse_hh_pair(chosen: str, rejected: str):
    """Parse the shared history, then append the chosen completion as one assistant turn.

    A completion can itself contain literal "Human:"/"Assistant:" text, which is generated text
    rather than further user messages. The pair diverges after the last shared assistant marker.
    """
    marker = "\n\nAssistant:"
    boundary = commonprefix([chosen, rejected]).rfind(marker)
    if boundary < 0:
        raise ValueError("HH preference pair has no shared assistant response boundary")
    return _parse_hh(chosen[:boundary]) + [{"role": "assistant",
                                            "content": chosen[boundary + len(marker):].strip()}]


def load_hh_rlhf():
    for sub in ["helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base"]:
        for split in ["train", "test"]:
            with gzip.open(hf_file("Anthropic/hh-rlhf", f"{sub}/{split}.jsonl.gz"), "rt", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    d = json.loads(line)
                    cid = f"{sub}/{split}/{i}"
                    yield {"conv_id": cid, "group_id": cid, "model": None, "timestamp": None, "lang_label": None,
                           "presplit": split, "conversation": _parse_hh_pair(d["chosen"], d["rejected"]),
                           "meta": {"subset": sub}}


# --------------------------------------------------------------------------- #
# ShareChat: turn-level CSVs grouped by URL.
# --------------------------------------------------------------------------- #
SHARECHAT_USER_ROLES = {"user", "human"}
SHARECHAT_ASSISTANT_ROLES = {"llm", "assistant", "model", "chatbot", "ai"}


def sharechat_files():
    return [hf_file("tucnguyen/ShareChat", f"{p}_results_final_language_filtered.csv") for p in SHARECHAT_PLATFORMS]


def _sharechat_metadata(rows, platform):
    """Language from every user row (mixed-language conversations get no label); model from assistant rows."""
    users = [r for r in rows if (r.get("role") or "").lower() in SHARECHAT_USER_ROLES]
    assistants = [r for r in rows if (r.get("role") or "").lower() in SHARECHAT_ASSISTANT_ROLES]
    languages = sorted({r.get("detected_language_final") or "" for r in users})
    if users and all(label in ENGLISH_LABELS for label in languages):
        language = "English"
    elif len(languages) == 1:
        language = languages[0]
    else:
        language = None
    model = next((r.get("model") for r in assistants if r.get("model")), "")
    timestamp = next((rows[0].get(k) for k in ["created_at", "create_time", "message_create_time", "published_at"]
                      if rows[0].get(k)), None)
    return language, f"{platform}:{model}", timestamp, languages


def load_sharechat():
    csv.field_size_limit(1 << 30)
    for platform, path in zip(SHARECHAT_PLATFORMS, sharechat_files()):
        convs = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for d in csv.DictReader(f):
                if d.get("url"):
                    convs.setdefault(d["url"], []).append(d)
        for url, rows in convs.items():
            rows = sorted(rows, key=lambda r: int(r.get("message_index") or 0))
            conv = _turns([{"role": (r.get("role") or "").lower(), "content": r.get("plain_text")} for r in rows],
                          user_roles=SHARECHAT_USER_ROLES, assistant_roles=SHARECHAT_ASSISTANT_ROLES)
            lang, model, timestamp, user_languages = _sharechat_metadata(rows, platform)
            yield {"conv_id": url, "group_id": url, "model": model, "timestamp": timestamp, "lang_label": lang,
                   "presplit": None, "conversation": conv,
                   "meta": {"platform": platform, "user_language_labels": user_languages}}
        del convs


def sharechat_ambiguous_order(paths):
    """URLs with repeated message indices: their message order cannot be reconstructed."""
    csv.field_size_limit(1 << 30)
    known, ambiguous = set(), set()
    for path in paths:
        seen = {}
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                url = row["url"]
                if not url:
                    continue
                indices = seen.setdefault(url, set())
                if row["message_index"] in indices:
                    ambiguous.add(url)
                indices.add(row["message_index"])
        known.update(seen)
    return known, ambiguous


# --------------------------------------------------------------------------- #
# Arena human preference 140K (2025): one longest history per evaluation session.
# --------------------------------------------------------------------------- #
def _arena_message(message, role):
    if not message or message.get("role") != role:
        return "", False
    blocks = message.get("content")
    if not isinstance(blocks, list) or not blocks:
        return "", False
    valid = all(b.get("type") == "text" and isinstance(b.get("text"), str) and not b.get("image") for b in blocks)
    text = "".join(b.get("text") or "" for b in blocks)
    return text, valid and bool(text.strip())


def load_arena_2025(stats=None):
    """Keep the longest released history of each session; the assistant text is the displayed
    left-side (model A) view. Sessions whose label is not English or with non-text content are dropped."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    repo = "lmarena-ai/arena-human-preference-140k"
    snapshot = hf_snapshot(repo)
    stats = Counter() if stats is None else stats
    best, session_order = {}, {}
    for path in sorted(snapshot.glob("data/*.parquet")):
        table = pq.read_table(path, columns=["id", "evaluation_session_id", "evaluation_order", "timestamp",
                                             "full_conversation"])
        lengths = pc.list_value_length(table["full_conversation"]).to_pylist()
        for row, length in zip(table.drop(["full_conversation"]).to_pylist(), lengths):
            sid = row["evaluation_session_id"]
            session_order.setdefault(sid, len(session_order))
            rank = (length, row["evaluation_order"], str(row["timestamp"]), row["id"])
            if sid not in best or rank > best[sid]:
                best[sid] = rank
        del table
    wanted = {rank[-1] for rank in best.values()}
    records = []
    for path in sorted(snapshot.glob("data/*.parquet")):
        cols = ["id", "evaluation_session_id", "evaluation_order", "timestamp", "language", "winner",
                "model_a", "model_b", "full_conversation"]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=cols):
            for row in batch.to_pylist():
                if row["id"] not in wanted:
                    continue
                if row["language"] != "en":
                    stats["excluded_non_english_session_label"] += 1
                    continue
                conv, valid = [], bool(row["full_conversation"])
                for turn in row["full_conversation"]:
                    user, vu = _arena_message(turn["user"], "user")
                    a, va = _arena_message(turn["model_side_a"], "assistant")
                    _, vb = _arena_message(turn["model_side_b"], "assistant")
                    valid &= vu and va and vb
                    conv.extend([{"role": "user", "content": user}, {"role": "assistant", "content": a}])
                if not valid:
                    stats["excluded_invalid_text_structure"] += 1
                    continue
                records.append({"conv_id": row["id"], "group_id": row["evaluation_session_id"], "model": None,
                                "timestamp": str(row["timestamp"]), "lang_label": "en", "presplit": None,
                                "conversation": conv,
                                "meta": {"evaluation_order": row["evaluation_order"], "winner": row["winner"],
                                         "model_a": row["model_a"], "model_b": row["model_b"],
                                         "conversation_view": "full_history_displayed_side_a"}})
    records.sort(key=lambda r: session_order[r["group_id"]])
    yield from records


LOADERS = {
    "wildchat_1m": load_wildchat_1m,
    "wildchat_4p8m": load_wildchat_4p8m,
    "lmsys": load_lmsys,
    "arena_2025": load_arena_2025,
    "sharechat": load_sharechat,
    "sharegpt": load_sharegpt,
    "hh_rlhf": load_hh_rlhf,
}
