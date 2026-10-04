"""Annotate conversations with three facets of the AI Observatory taxonomy (Sec. 2.3, App. C.1).

Requires the AI Observatory package (`naturalistic_ai`, Longpre et al., 2026), installed separately.
Prompts, label sets, the label hierarchy, conversation formatting, and the acceptance rule all come from
that package and its default configuration (config/default.yaml); the annotation model is gpt-5.6-luna
with medium reasoning effort. Every user prompt within the configured message cap is annotated:

  function   function_purpose, one annotation per user prompt, aggregated to nine broad categories
  topic      topic, one annotation per user prompt + assistant response, nine broad categories
  multiturn  multi_turn_relationship of every follow-up prompt (four categories); the first prompt of
             a conversation is "First request" by position

An annotation is accepted when its confidence reaches the configured threshold (0.7). A conversation's
facet is complete when all of its units were accepted; its category vector marks the broad categories
present in any unit.

    OPENAI_API_KEY=... python -m src.taxonomy.annotate --source wildchat_1m --split train --facet function

Output: artifacts/results/taxonomy/annotations/<source>_<split>_<facet>/
  units.jsonl          one record per annotated unit (resumable; failed units are retried with --retry_failed)
  conversations.jsonl  per-conversation categories, category vector, and completeness
"""
import argparse
import asyncio
import json
import os
from collections import Counter, defaultdict
from importlib import resources

import yaml

from src.config import CORPUS, RESULTS, SPLITS, TAXONOMY_SOURCES

MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "medium"
FACETS = {
    "function": {"level": "prompt", "task": "function_purpose", "hierarchy": "function_purpose_hierarchy"},
    "topic": {"level": "turn", "task": "topic", "hierarchy": "topic_hierarchy"},
    "multiturn": {"level": "prompt", "task": "multi_turn_relationship", "hierarchy": None},
}
FIRST_REQUEST = "First request"
OUT = RESULTS / "taxonomy" / "annotations"


def upstream_config():
    root = resources.files("naturalistic_ai").joinpath("config")
    load = lambda name: yaml.safe_load(root.joinpath(name).read_text())
    return load("default.yaml"), load("instructions.yaml"), load("taxonomy_options.yaml"), load("label_metadata.yaml")


def setup(facet):
    from naturalistic_ai import models
    from naturalistic_ai.pipeline.processor import AnnotationProcessor
    defaults, instructions, options, metadata = upstream_config()
    spec = FACETS[facet]
    processor = AnnotationProcessor(
        instructions, options, instruction_first=defaults["instruction_first"], multi_hist=defaults["multi_hist"],
        max_prev_chars=defaults["max_prev_chars"], max_turns=defaults["max_turns"],
        max_assistant_content_chars=defaults["max_assistant_content_chars"])
    labels = [x.split(":", 1)[0].strip() for x in options[spec["level"]][spec["task"]]]
    hierarchy = metadata[spec["hierarchy"]] if spec["hierarchy"] else {label: [label] for label in labels}
    if facet == "multiturn":
        hierarchy = {k: v for k, v in hierarchy.items() if k != FIRST_REQUEST}
    return models, processor, labels, hierarchy, models.AnnotationLevel(spec["level"]), defaults


def build_units(rows, models, processor, facet, level, defaults):
    """One prompt per annotation unit, enumerating every user prompt within the message cap."""
    task, units = FACETS[facet]["task"], []
    for row in rows:
        messages = row["conversation"]
        if level == models.AnnotationLevel.PROMPT and len(messages) % 2:
            # A final unanswered user prompt is still a prompt-level unit; the empty reply is never shown.
            messages = messages + [{"role": "assistant", "content": ""}]
        conv = models.Conversation(conversation_id=row["conv_id"], dataset_id="corpus", conversation=[
            models.Message(turn=i, role=m["role"], content=m["content"]) for i, m in enumerate(messages)])
        pairs = len(messages) // 2
        if defaults["max_turns"]:
            pairs = min(pairs, defaults["max_turns"] // 2)
        for index in range(pairs):
            unit = {"conversation_id": row["conv_id"], "turn": index}
            if facet == "multiturn" and index == 0:
                unit["labels"], unit["status"] = [FIRST_REQUEST], "accepted"
            else:
                unit["prompt"] = processor.build_prompt(conv, level, task, turn_index=index, include_prev_turn=True,
                                                        format_type=defaults["format_type"], output_mode="structured")
            units.append(unit)
    return units


def response_schema(models, labels, facet):
    from openai.lib._pydantic import to_strict_json_schema
    schema = to_strict_json_schema(models.StructuredAnnotationResponse)
    item = schema["$defs"]["StructuredAnnotationItem"]["properties"]
    item["confidence"].pop("default", None)
    item["labels"]["items"]["enum"] = [x for x in labels if facet != "multiturn" or x != FIRST_REQUEST]
    if facet == "multiturn":   # a single relationship per follow-up prompt
        schema["properties"]["annotations"]["maxItems"] = 1
        item["labels"]["maxItems"] = 1
    return schema


async def annotate(units, out_path, models, labels, facet, level, threshold, concurrency, schema):
    import openai
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=180, max_retries=0)
    queue = asyncio.Queue()
    for u in units:
        queue.put_nowait(u)
    fatal = asyncio.Event()
    with out_path.open("a", buffering=1) as f:
        async def worker():
            while not queue.empty() and not fatal.is_set():
                unit = queue.get_nowait()
                record = {k: v for k, v in unit.items() if k != "prompt"}
                for attempt in range(3):
                    try:
                        r = await client.responses.create(
                            model=MODEL, reasoning={"effort": REASONING_EFFORT}, input=unit["prompt"], store=False,
                            text={"format": {"type": "json_schema", "name": f"{facet}_annotation", "strict": True,
                                             "schema": schema}},
                            max_output_tokens=4096 if attempt == 0 else 8192)
                        if r.status != "completed" or not r.output_text:
                            record.update(status="incomplete", labels=[])
                        else:
                            parsed = models.StructuredAnnotationResponse.model_validate_json(r.output_text)
                            result = parsed.to_annotation_result(unit["conversation_id"], level,
                                                                 FACETS[facet]["task"], unit["turn"])
                            raw = [x for item in parsed.annotations for x in item.labels]
                            explicit_none = bool(raw) and all(x.strip().lower() == "none" for x in raw)
                            if not set(result.labels) <= set(labels) or not (result.labels or explicit_none):
                                raise ValueError("labels outside the taxonomy")
                            record.update(labels=result.labels, confidence=result.confidence,
                                          status="accepted" if result.confidence >= threshold else "low_confidence")
                    except openai.APIStatusError as exc:
                        record.update(status="error", labels=[], http_status=exc.status_code)
                        if exc.status_code in {400, 401, 403, 404}:
                            fatal.set()
                    except Exception as exc:
                        record.update(status="invalid" if isinstance(exc, ValueError) else "error", labels=[],
                                      error_type=type(exc).__name__)
                    if record["status"] not in {"error", "invalid", "incomplete"} or fatal.is_set():
                        break
                    await asyncio.sleep(2 ** (attempt + 1))
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        try:
            await asyncio.gather(*(worker() for _ in range(concurrency)))
        finally:
            await client.close()
    if fatal.is_set():
        raise SystemExit("the API rejected the request configuration")


def summarize(rows, units, latest, hierarchy, out):
    expected = defaultdict(list)
    for u in units:
        expected[u["conversation_id"]].append((u["conversation_id"], u["turn"]))
    parents = {child: parent for parent, children in hierarchy.items() for child in children}
    with (out / "conversations.jsonl").open("w") as f:
        for row in rows:
            records = [latest.get(k, {"status": "pending"}) for k in expected[row["conv_id"]]]
            accepted = [r for r in records if r["status"] == "accepted"]
            broad = sorted({parents[x] for r in accepted for x in r["labels"] if x in parents})
            f.write(json.dumps({"conversation_id": row["conv_id"], "complete": len(accepted) == len(records) > 0,
                                "categories": broad, "category_order": list(hierarchy),
                                "vector": [int(p in broad) for p in hierarchy]}) + "\n")
    print(dict(Counter(r["status"] for r in latest.values())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=TAXONOMY_SOURCES, required=True)
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--facet", choices=list(FACETS), required=True)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--retry_failed", action="store_true")
    ap.add_argument("--prepare_only", action="store_true", help="build prompts and count units without API calls")
    args = ap.parse_args()
    models, processor, labels, hierarchy, level, defaults = setup(args.facet)
    rows = [json.loads(line) for line in (CORPUS / args.source / f"{args.split}.jsonl").open()]
    units = build_units(rows, models, processor, args.facet, level, defaults)
    out = OUT / f"{args.source}_{args.split}_{args.facet}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "units.jsonl"
    latest = {}
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            latest[(r["conversation_id"], r["turn"])] = r
    with path.open("a") as f:     # first requests are labeled by position
        for u in units:
            if "prompt" not in u and (u["conversation_id"], u["turn"]) not in latest:
                f.write(json.dumps(u) + "\n")
                latest[(u["conversation_id"], u["turn"])] = u
    retry = {"error", "invalid", "incomplete"} if args.retry_failed else set()
    pending = [u for u in units if "prompt" in u and ((u["conversation_id"], u["turn"]) not in latest
                                                      or latest[(u["conversation_id"], u["turn"])]["status"] in retry)]
    print(f"{len(rows)} conversations, {len(units)} units, {len(pending)} pending", flush=True)
    if pending and not args.prepare_only:
        asyncio.run(annotate(pending, path, models, labels, args.facet, level, defaults["confidence_threshold"],
                             args.concurrency, response_schema(models, labels, args.facet)))
        for line in path.open():
            r = json.loads(line)
            latest[(r["conversation_id"], r["turn"])] = r
    summarize(rows, units, latest, hierarchy, out)


if __name__ == "__main__":
    main()
