"""Simulate task conversations between a user model and an assistant (Sec. 3.2).

Each problem is the user model's intent; the assistant never sees it. User-model generation guardrails
of Naous et al. (2026): <|endconversation|> is disabled (logit bias -100), a turn may not start with
"I", "You", or "Here" (either case), must contain 3-25 words, and may not copy the problem or an earlier
user turn; a violating reply is resampled (up to 32 times). User sampling: temperature 1.0, top-p 1.0.
Conversations last eight assistant turns. Five conversations per problem.

    vllm serve artifacts/models/userlm_wildchat_1m_meta-llama--Meta-Llama-3-8B_hf --port 8001 --max-model-len 8192
    vllm serve Qwen/Qwen2.5-14B-Instruct --port 8002
    python -m src.task_sim.simulate --user_source wildchat_1m \
        --user_model_dir artifacts/models/userlm_wildchat_1m_meta-llama--Meta-Llama-3-8B_hf \
        --assistant Qwen/Qwen2.5-14B-Instruct

Output: artifacts/data/task_conversations/<assistant tag>/<user_source>/{gsm8k,humaneval}.jsonl
(resumable; conversations interrupted by engine errors are not written and are retried on rerun).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.config import DATA, SOURCES, USER_BASE_MODELS
from src.task_sim.tasks import DATASETS, TASKS
from src.userlm.clients import AssistantClient, UserModelClient, make_client, served_model

OUT = DATA / "task_conversations"
ASSISTANT_MODELS = ["Qwen/Qwen2.5-14B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
BANNED_FIRST_WORDS = ("I", "You", "Here", "i", "you", "here")
MIN_WORDS, MAX_WORDS = 3, 25
MAX_TURNS = 8
SAMPLES = 5


def assistant_tag(model):
    return model.split("/")[-1].lower().replace(".", "p")


def normalize(text):
    return " ".join(text.lower().split())


def make_validator(tokenizer, task):
    banned = set()
    for word in BANNED_FIRST_WORDS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        assert len(ids) == 1, word
        banned.add(ids[0])
    copies = {normalize(task["problem"]), normalize(task["intent"])}

    def check(reply, conversation):
        ids = tokenizer.encode(reply, add_special_tokens=False)
        if ids and ids[0] in banned:
            return "first_token"
        n = len(reply.split())
        if n > MAX_WORDS:
            return "too_long"
        if n < MIN_WORDS:
            return "too_short"
        text = normalize(reply)
        if text in copies or any(text == normalize(t["content"]) for t in conversation if t["role"] == "user"):
            return "verbatim"
        return None
    return check


def run_conversation(task, k, user, assistant):
    validator = make_validator(user.tok, task)
    conv = []
    turn = user.generate(task["intent"], conv, allow_terminal=False, validator=validator)
    if turn.terminal:
        return None
    while True:
        conv.append({"role": "user", "content": turn.text})
        reply = assistant.generate(conv)
        if reply.terminal:          # engine failure: retried on the next run
            return None
        conv.append({"role": "assistant", "content": reply.text})
        if len(conv) // 2 >= MAX_TURNS:
            break
        turn = user.generate(task["intent"], conv, allow_terminal=False, validator=validator)
        if turn.terminal:           # e.g. the user model's context is full
            if turn.reason == "error":
                return None
            break
    return {"conv_id": f"{task['task_id']}#{k}", "task_id": task["task_id"], "dataset": task["dataset"],
            "sample_index": k, "conversation": conv}


def main():
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user_source", choices=SOURCES, required=True)
    ap.add_argument("--user_model_dir", required=True)
    ap.add_argument("--user_base_model", choices=USER_BASE_MODELS, default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--assistant", choices=ASSISTANT_MODELS, required=True)
    ap.add_argument("--user_port", type=int, default=8001)
    ap.add_argument("--assistant_ports", type=int, nargs="+", default=[8002])
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    user_client = make_client(f"http://localhost:{args.user_port}/v1", args.workers + 8)
    tok = AutoTokenizer.from_pretrained(args.user_model_dir)
    user = UserModelClient(user_client, served_model(user_client), tok, args.user_base_model, temperature=1.0,
                           top_p=1.0, max_model_len=8192, max_resamples=32, temperature_boost=0.0,
                           logit_bias={tok.convert_tokens_to_ids("<|endconversation|>"): -100})
    clients = [make_client(f"http://localhost:{p}/v1", args.workers + 8, retries=0) for p in args.assistant_ports]
    assistant = AssistantClient(clients, args.assistant, temperature=0.7, top_p=0.8, top_k=20, max_tokens=4096)
    out = OUT / assistant_tag(args.assistant) / args.user_source
    out.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        tasks = [json.loads(line) for line in (TASKS / f"{dataset}.jsonl").open()]
        path = out / f"{dataset}.jsonl"
        done = {json.loads(line)["conv_id"] for line in path.open()} if path.exists() else set()
        todo = [(t, k) for t in tasks for k in range(SAMPLES) if f"{t['task_id']}#{k}" not in done]
        failed = 0
        with ThreadPoolExecutor(args.workers) as ex, path.open("a") as f:
            futures = [ex.submit(run_conversation, t, k, user, assistant) for t, k in todo]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{args.user_source}/{dataset}"):
                record = fut.result()
                if record is None:
                    failed += 1
                else:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{dataset}: {len(todo) - failed} written, {failed} failed (rerun to retry)")


if __name__ == "__main__":
    main()
