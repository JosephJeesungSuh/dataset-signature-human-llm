"""Generate synthetic conversations between a user model and an assistant (Sec. 3.1, App. D-E).

The user model (served by vLLM, raw completions) is conditioned on intents of a third dataset and
generates the first request itself; the assistant (served by vLLM, chat completions, no system prompt)
never sees the intent. A conversation ends when the user emits <|endconversation|>, after four assistant
replies, or at the user model's 8,192-token context. One conversation per intent; splits follow the
intent's original split. User sampling: temperature 0.7, top-p 0.9; assistant sampling: temperature 0.7,
top-p 0.8, top-k 20, thinking disabled.

    vllm serve artifacts/models/userlm_lmsys_meta-llama--Meta-Llama-3-8B_hf --port 8002 --max-model-len 8192
    vllm serve Qwen/Qwen3.5-9B --port 8001
    python -m src.userlm.synth_conversations --intent_source wildchat_4p8m --user_source lmsys \
        --user_model_dir artifacts/models/userlm_lmsys_meta-llama--Meta-Llama-3-8B_hf --assistant qwen3p5-9b

Output: artifacts/data/synthetic/<name>/{train,val,test}.jsonl in the corpus record format. Generation is
resumable; conversations that failed (engine errors) are not written, so rerunning retries them.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.config import ASSISTANTS, CORPUS, SOURCES, SPLITS, SYNTHETIC, USER_BASE_MODELS, USERLM_DATA, synthetic_name
from src.curation.common import conversation_metadata, validate_conversation
from src.userlm.clients import AssistantClient, UserModelClient, make_client, served_model

MAX_ASSISTANT_TURNS = 4
USER_CONTEXT = 8192


def run_conversation(seed, user, assistant, max_turns):
    intent, conv = seed["intent"].strip(), []
    turn = user.generate(intent, conv, allow_terminal=False)
    if turn.terminal:
        return None, f"first_turn_{turn.reason}"
    while True:
        conv.append({"role": "user", "content": turn.text})
        reply = assistant.generate(conv)
        if reply.terminal:
            conv.pop()
            if not conv:
                return None, f"assistant_{reply.reason}"
            ended_by = f"assistant_{reply.reason}"
            break
        conv.append({"role": "assistant", "content": reply.text})
        if len(conv) // 2 >= max_turns:
            ended_by = "max_turns"
            break
        turn = user.generate(intent, conv)
        if turn.terminal:
            ended_by = turn.reason
            break
    if validate_conversation(conv) is not None:
        return None, "invalid"
    return conv, ended_by


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intent_source", choices=SOURCES, required=True)
    ap.add_argument("--user_source", choices=SOURCES, required=True, help="training dataset of the user model")
    ap.add_argument("--user_model_dir", required=True, help="converted user model (also provides the tokenizer)")
    ap.add_argument("--user_base_model", choices=USER_BASE_MODELS, default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--assistant", choices=list(ASSISTANTS), required=True)
    ap.add_argument("--user_port", type=int, default=8002)
    ap.add_argument("--assistant_ports", type=int, nargs="+", default=[8001])
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--max_turns", type=int, default=MAX_ASSISTANT_TURNS)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    if args.intent_source == args.user_source:
        ap.error("the intent source must differ from the user model's training dataset")
    user_client = make_client(f"http://localhost:{args.user_port}/v1", args.workers + 8)
    user = UserModelClient(user_client, served_model(user_client), AutoTokenizer.from_pretrained(args.user_model_dir),
                           args.user_base_model, temperature=0.7, top_p=0.9, max_model_len=USER_CONTEXT)
    clients = [make_client(f"http://localhost:{p}/v1", args.workers + 8, retries=0) for p in args.assistant_ports]
    assistant = AssistantClient(clients, ASSISTANTS[args.assistant], temperature=0.7, top_p=0.8, top_k=20,
                                enable_thinking=False)
    name = synthetic_name(args.intent_source, args.user_source, args.assistant, args.user_base_model)
    out_dir = SYNTHETIC / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        intents = {}
        for line in (USERLM_DATA / args.intent_source / "intents" / f"{split}.jsonl").open():
            r = json.loads(line)
            intents[r["conv_id"]] = r
        order = [json.loads(line)["conv_id"] for line in (CORPUS / args.intent_source / f"{split}.jsonl").open()]
        path = out_dir / f"{split}.jsonl"
        done = {json.loads(line)["conv_id"] for line in path.open()} if path.exists() else set()
        seeds = [intents[c] for c in order if c in intents and c not in done]
        failures, t0 = {}, time.time()
        with ThreadPoolExecutor(args.workers) as ex, path.open("a", encoding="utf-8") as f:
            futures = {ex.submit(run_conversation, s, user, assistant, args.max_turns): s for s in seeds}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{name}/{split}"):
                seed, (conv, ended_by) = futures[fut], fut.result()
                if conv is None:
                    failures[ended_by] = failures.get(ended_by, 0) + 1
                    continue
                record = {"conv_id": seed["conv_id"], "group_id": seed["conv_id"], "source": name,
                          "conversation": conv, **conversation_metadata(conv),
                          "meta": {"synthetic": True, "seed_conv_id": seed["conv_id"],
                                   "intent_source": args.intent_source, "intent": seed["intent"],
                                   "user_source": args.user_source, "user_model": args.user_model_dir,
                                   "assistant_model": ASSISTANTS[args.assistant], "ended_by": ended_by}}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{name}/{split}: {len(seeds) - sum(failures.values())} written, failures {failures} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
