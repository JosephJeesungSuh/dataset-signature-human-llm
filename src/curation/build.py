"""Stage 2 of curation: deduplicate, filter templates, delete markers, and split all seven sources jointly.

Per source (App. A.2): exact duplicates of the concatenated user text are removed (lowercased, whitespace
collapsed); opening prompts containing a 7-gram that occurs more than 100 times in the source are removed;
WildChat-4.8M drops conversation hashes present in WildChat-1M; ShareChat drops conversations with mixed
user languages or ambiguous message order. Markers (markers.py) are deleted, and conversations that become
empty or duplicated are removed. Full user texts of at least five words that occur in several sources are
removed everywhere.

Records sharing a user (WildChat users are shared across releases), a full user text, or an opening prompt
of at least five words form connected groups that never cross splits. HH-RLHF keeps its released test
membership. Each source then contributes exactly 40,000 / 3,000 / 7,000 train / validation / test
conversations; a shortage is an error. ShareGPT, with fewer than 50,000 eligible conversations, keeps all of
them, split in the same 40:3:7 proportions by whole groups.

Two aligned copies are written with identical IDs and splits:
    artifacts/data/corpus/<source>/{train,val,test}.jsonl            markers deleted (main input)
    artifacts/data/corpus_original/<source>/{train,val,test}.jsonl   markers kept ("restoring markers")

    python -m src.curation.build
"""
import argparse
import gc
import json
import random
import time
from collections import Counter, defaultdict

from src.config import CORPUS, CORPUS_ORIGINAL, INTERIM, SEED, SOURCES, SPLIT_SIZES, SPLITS
from src.curation.common import (MIN_WORDS_FOR_SHARING, NGRAM_THRESHOLD, extract_ngrams, first_user_turn,
                                 read_jsonl, sha256, validate_conversation, write_jsonl)
from src.curation.markers import clean_record
from src.curation.sources import (ENGLISH_LABELS, WILDCHAT_SUPERSEDED, sharechat_ambiguous_order,
                                  sharechat_files)


class UnionFind:
    def __init__(self):
        self.parent, self.size = [], []

    def add(self):
        self.parent.append(len(self.parent))
        self.size.append(1)
        return len(self.parent) - 1

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.size[a] < self.size[b]:
                a, b = b, a
            self.parent[b] = a
            self.size[a] += self.size[b]


def allowed_splits(source, presplit):
    """HH-RLHF's released test conversations stay in test; its training conversations never enter test."""
    if source == "hh_rlhf":
        return {"test"} if presplit == "test" else {"train", "val"}
    return set(SPLITS)


def filter_templates(records):
    """Drop records whose opening prompt contains a 7-gram occurring more than NGRAM_THRESHOLD times."""
    counts = Counter()
    for r in records:
        counts.update(extract_ngrams(first_user_turn(r["conversation"])))
    return [r for r in records
            if not any(counts[g] > NGRAM_THRESHOLD for g in extract_ngrams(first_user_turn(r["conversation"])))]


# Sources that keep every eligible conversation (fewer than 50,000), split 40:3:7 by whole groups.
ALL_ELIGIBLE = {"sharegpt"}


def split_caps(source, n_eligible):
    if source not in ALL_ELIGIBLE:
        return dict(SPLIT_SIZES)
    total = sum(SPLIT_SIZES.values())
    caps = {split: round(n_eligible * SPLIT_SIZES[split] / total) for split in ("val", "test")}
    caps["train"] = n_eligible - caps["val"] - caps["test"]
    return caps


def group_namespace(source):
    return "wildchat" if source.startswith("wildchat") else source


def filter_sources(work, manifest):
    """Write each source's filtered records to `work` and return user texts shared across sources."""
    owners = defaultdict(set)
    sharechat_known = sharechat_ambiguous = None
    for src in SOURCES:
        start, st = time.time(), Counter()
        earlier = set()
        if src in WILDCHAT_SUPERSEDED:
            earlier = {r["conv_id"] for r in read_jsonl(INTERIM / f"{WILDCHAT_SUPERSEDED[src]}.jsonl")}
        if src == "sharechat":
            sharechat_known, sharechat_ambiguous = sharechat_ambiguous_order(sharechat_files())
        records, seen = [], set()
        for r in read_jsonl(INTERIM / f"{src}.jsonl"):
            st["english"] += 1
            assert r["source"] == src and validate_conversation(r["conversation"]) is None
            if src == "hh_rlhf":
                assert r["presplit"] in {"train", "test"}
            else:
                r["presplit"] = None
            if r["conv_id"] in earlier:
                st["excluded_in_earlier_release"] += 1
                continue
            if src == "sharechat":
                if r["conv_id"] not in sharechat_known:
                    raise ValueError(f"ShareChat conversation missing from the source CSVs: {r['conv_id']}")
                labels = r["meta"].get("user_language_labels")
                if not labels or not all(label in ENGLISH_LABELS for label in labels):
                    st["excluded_mixed_user_language"] += 1
                    continue
                if r["conv_id"] in sharechat_ambiguous:
                    st["excluded_ambiguous_message_order"] += 1
                    continue
            if r["user_text_hash"] in seen:
                continue
            seen.add(r["user_text_hash"])
            records.append(r)
        st["after_exact_dedup"] = len(records)
        records = filter_templates(records)
        st["after_template_filter"] = len(records)
        kept, clean_seen = [], set()
        for r in records:
            c = clean_record(r)
            if validate_conversation(c["conversation"]) is not None:
                st["excluded_empty_after_marker_deletion"] += 1
                continue
            if c["user_text_hash"] in clean_seen:
                st["excluded_duplicate_after_marker_deletion"] += 1
                continue
            clean_seen.add(c["user_text_hash"])
            kept.append(r)
            if c["n_words_user"] >= MIN_WORDS_FOR_SHARING:
                owners[c["user_text_hash"]].add(src)
        write_jsonl(work / f"{src}.jsonl", kept)
        manifest["sources"][src] = dict(st)
        del records, kept, seen, clean_seen, earlier
        gc.collect()
        print(f"filtered {src}: {dict(st)} ({time.time() - start:.0f}s)", flush=True)
    shared = {h for h, s in owners.items() if len(s) > 1}
    manifest["cross_source_user_texts_removed"] = len(shared)
    return shared


def assign(work, shared, manifest, rng):
    """Connected-group split assignment, then exact per-split sampling within the assigned groups."""
    uf, key_owner, rows = UnionFind(), {}, []
    for src in SOURCES:
        st = manifest["sources"][src]
        st["eligible"] = 0
        for line, r in enumerate(read_jsonl(work / f"{src}.jsonl")):
            c = clean_record(r)
            if c["user_text_hash"] in shared:
                continue
            st["eligible"] += 1
            i = uf.add()
            keys = [("group", group_namespace(src), str(r["group_id"])), ("user_text", c["user_text_hash"])]
            if len(first_user_turn(c["conversation"]).split()) >= MIN_WORDS_FOR_SHARING:
                keys.append(("first_turn", c["first_turn_hash"]))
            for key in keys:
                if key in key_owner:
                    uf.union(i, key_owner[key])
                else:
                    key_owner[key] = i
            rows.append((src, line, r.get("presplit")))
    del key_owner
    components = defaultdict(list)
    for i in range(len(rows)):
        components[uf.find(i)].append(i)
    del uf
    permitted, assignment = {}, {}
    by_source = {s: defaultdict(list) for s in SOURCES}
    conflicts = Counter()
    for gid, members in components.items():
        restrictions = [allowed_splits(rows[i][0], rows[i][2]) for i in members]
        common = set.intersection(*restrictions)
        if not common:
            # Keep the released HH-RLHF test side of a train/test conflict; drop the conflicting records.
            conflicts["components_with_conflicting_presplits"] += 1
            common = {"test"}
        permitted[gid] = common
        if len(common) == 1:
            assignment[gid] = next(iter(common))
        for i, allowed in zip(members, restrictions):
            src, line, _ = rows[i]
            if allowed & common:
                by_source[src][gid].append(line)
            else:
                conflicts[f"{src}:excluded_presplit_conflict"] += 1
    manifest["presplit_conflicts"] = dict(conflicts)
    del components, rows
    selected = {}
    # ShareGPT has the fewest eligible conversations, so its groups are placed first.
    for src in ["sharegpt"] + [s for s in SOURCES if s != "sharegpt"]:
        groups = by_source[src]
        caps = split_caps(src, sum(map(len, groups.values())))
        gids = sorted(groups)
        rng.shuffle(gids)
        for split in ["test", "val"]:
            count = sum(len(groups[g]) for g in gids if assignment.get(g) == split)
            for g in gids:
                if count >= caps[split]:
                    break
                if g not in assignment and split in permitted[g]:
                    assignment[g] = split
                    count += len(groups[g])
        for g in gids:
            if g not in assignment:
                assert "train" in permitted[g]
                assignment[g] = "train"
        chosen, pools = {}, {}
        for split in SPLITS:
            lines = [i for g in gids if assignment[g] == split for i in groups[g]]
            pools[split] = len(lines)
            if src not in ALL_ELIGIBLE:
                if len(lines) < caps[split]:
                    raise ValueError(f"{src}/{split}: only {len(lines)} conversations for {caps[split]}")
                rng.shuffle(lines)
                lines = lines[:caps[split]]
            chosen.update({i: split for i in lines})
        selected[src] = chosen
        manifest["sources"][src].update(assigned_pools=pools, counts=dict(Counter(chosen.values())))
        print(f"assigned {src}: pools {pools}", flush=True)
    return selected


def validate(root):
    """Structure, uniqueness, exact counts, and split/source isolation of a written corpus."""
    seen = {k: {} for k in ["group", "user_text", "first_turn"]}
    owner, ids_by_source = {}, {}
    manifest = json.loads((root / "manifest.json").read_text())
    for src in SOURCES:
        ids, texts = set(), set()
        for split in SPLITS:
            n = 0
            for r in read_jsonl(root / src / f"{split}.jsonl"):
                n += 1
                assert r["source"] == src and validate_conversation(r["conversation"]) is None
                assert r["conv_id"] not in ids and r["user_text_hash"] not in texts
                ids.add(r["conv_id"])
                texts.add(r["user_text_hash"])
                assert split in allowed_splits(src, r.get("presplit"))
                if r["n_words_user"] >= MIN_WORDS_FOR_SHARING:
                    assert owner.setdefault(r["user_text_hash"], src) == src, "user text shared across sources"
                keys = {"group": (group_namespace(src), str(r["group_id"])), "user_text": r["user_text_hash"]}
                if len(first_user_turn(r["conversation"]).split()) >= MIN_WORDS_FOR_SHARING:
                    keys["first_turn"] = r["first_turn_hash"]
                for name, key in keys.items():
                    assert seen[name].setdefault(key, split) == split, f"{name} crosses splits"
            expected = manifest["sources"][src]["counts"][split] if src in ALL_ELIGIBLE else SPLIT_SIZES[split]
            assert n == expected, (src, split, n)
        ids_by_source[src] = ids
    for later, earlier in WILDCHAT_SUPERSEDED.items():
        assert not ids_by_source[later] & ids_by_source[earlier]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    for out in (CORPUS, CORPUS_ORIGINAL):
        if any((out / s).exists() for s in SOURCES):
            raise FileExistsError(f"{out} already contains corpus files")
    missing = [s for s in SOURCES if not (INTERIM / f"{s}.jsonl").is_file()]
    if missing:
        raise FileNotFoundError(f"Run src.curation.load first for: {missing}")
    work = CORPUS.parent / ".corpus_building"
    work.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    manifest = {"seed": args.seed, "split_sizes": SPLIT_SIZES, "sources": {}}
    shared = filter_sources(work, manifest)
    selected = assign(work, shared, manifest, rng)
    for src in SOURCES:
        buckets = {sp: [] for sp in SPLITS}
        for i, r in enumerate(read_jsonl(work / f"{src}.jsonl")):
            if i in selected[src]:
                buckets[selected[src][i]].append(r)
        for split, records in buckets.items():
            rng.shuffle(records)
            write_jsonl(CORPUS_ORIGINAL / src / f"{split}.jsonl", records)
            write_jsonl(CORPUS / src / f"{split}.jsonl", (clean_record(r) for r in records))
            for out in (CORPUS, CORPUS_ORIGINAL):
                manifest["sources"][src].setdefault(f"sha256_{out.name}", {})[split] = \
                    sha256(out / src / f"{split}.jsonl")
        (work / f"{src}.jsonl").unlink()
    work.rmdir()
    for root in (CORPUS, CORPUS_ORIGINAL):
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    validate(CORPUS)
    print("corpus written to", CORPUS, "and", CORPUS_ORIGINAL, flush=True)


if __name__ == "__main__":
    main()
