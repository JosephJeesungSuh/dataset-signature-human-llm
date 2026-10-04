"""Dataset-specific markers in user messages (Table 1) and their deletion.

Each marker is deleted and the whitespace around it is collapsed into a single space.
Assistant messages are left unchanged (they are masked for classification anyway).
"""
import re
from typing import Dict, List

from src.curation.common import conversation_metadata

# (name, pattern, literal hints used to skip the regex on most messages)
MARKERS = [
    ("numbered_name", r"\bNAME_\d+\b", ("NAME_",)),                                   # LMSYS
    ("your_answer", r"\[your answer\]", ("[your answer]",)),                          # LMSYS
    ("presidio", r"<PRESIDIO_ANONYMIZED_[A-Z_0-9]+>", ("<PRESIDIO_",)),               # WildChat PII redaction
    ("trufflehog", r"<TRUFFLEHOG_REDACTED_[A-Z_0-9]+>", ("<TRUFFLEHOG_",)),           # WildChat secret redaction
    ("angle_redacted", r"<REDACTED>", ("<REDACTED>",)),                               # ShareChat
    ("angle_url", r"<URL>", ("<URL>",)),                                              # ShareChat
    ("angle_date_time", r"<DATE_TIME>", ("<DATE_TIME>",)),                            # ShareChat
]
_COMPILED = [(name, re.compile(pattern), hints) for name, pattern, hints in MARKERS]


def find_markers(text: str) -> List[re.Match]:
    """Non-overlapping marker matches, leftmost first (earlier rules win ties)."""
    candidates = []
    for priority, (_, pattern, hints) in enumerate(_COMPILED):
        if any(h in text for h in hints):
            candidates.extend((m.start(), priority, m) for m in pattern.finditer(text))
    candidates.sort(key=lambda c: (c[0], c[1]))
    selected, end = [], 0
    for start, _, match in candidates:
        if start >= end:
            selected.append(match)
            end = match.end()
    return selected


def delete_markers(text: str) -> str:
    """Delete every marker; adjoining whitespace collapses into one space (none at the edges)."""
    while matches := find_markers(text):
        spans = []
        for m in matches:
            start, end = m.span()
            while start and text[start - 1].isspace():
                start -= 1
            while end < len(text) and text[end].isspace():
                end += 1
            if spans and start <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(end, spans[-1][1]))
            else:
                spans.append((start, end))
        parts, cursor = [], 0
        for start, end in spans:
            parts.append(text[cursor:start])
            if start and end < len(text):
                parts.append(" ")
            cursor = end
        parts.append(text[cursor:])
        text = "".join(parts)
    return text


def clean_record(record: Dict) -> Dict:
    """Copy of a corpus record with markers deleted from user messages and metadata recomputed."""
    conversation = [dict(t, content=delete_markers(t["content"])) if t["role"] == "user" else dict(t)
                    for t in record["conversation"]]
    return {**record, "conversation": conversation, **conversation_metadata(conversation)}


def has_marker(record: Dict) -> bool:
    return any(t["role"] == "user" and find_markers(t["content"]) for t in record["conversation"])
