"""Generate a high-level user intent for every conversation (App. D), following Naous et al. (2026).

The full conversation is given to Qwen3-32B (non-thinking) served by vLLM with an OpenAI-compatible API,
with the UserLM paper's three-shot intent prompt. Output (resumable):
    artifacts/userlm_data/<source>/intents/{split}.jsonl   (conv_id, conversation, intent)

    vllm serve Qwen/Qwen3-32B --port 8000 --max-model-len 40960
    python -m src.userlm.generate_intents --source wildchat_1m --ports 8000
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from src.config import CORPUS, INTENT_MODEL, SOURCES, SPLITS, USERLM_DATA

FEW_SHOT = [
    {"conversation": [
        {"role": "user", "content": "List to me 5 beautiful sights in Algeria"},
        {"role": "assistant", "content": "Algeria is a country of striking natural beauty and rich cultural heritage. Here are five breathtaking sights you should know about: Tassili n'Ajjer National Park, Djemila, Timgad, Beni Hammad Fort, Basilique Notre Dame d'Afrique"}],
     "intent": "You are a user chatting with an assistant language model to obtain some recommendations of places to see in Algeria."},
    {"conversation": [
        {"role": "user", "content": "does allertine cause weight gain"},
        {"role": "assistant", "content": "As an AI language model, I don't have personal experiences. But according to the available literature, Allertine is not known to cause weight gain. However, ..."},
        {"role": "user", "content": "do antihistamines cause weight gain"},
        {"role": "assistant", "content": "Most antihistamines do not typically cause weight gain as a direct side effect. However, some people find that they experience increased appetite or cravings for sugary, high-calorie ..."},
        {"role": "user", "content": "how much weight can i lose by cutting sugar for 40 days"},
        {"role": "assistant", "content": "The amount of weight you can lose by cutting sugar from your diet for 40 days will depend on a variety of factors, including your starting weight, age, gender, level of physical activity, ..."}],
     "intent": "You are a user chatting with an assistant language model to get information about strategies for losing weight and the impact of certain drugs or medications on weight gain."},
    {"conversation": [
        {"role": "user", "content": "Is einstein determinism challenged by quantum mechanics?"},
        {"role": "assistant", "content": "Quantum mechanics challenges Einstein's determinism, which held that the universe is fully predictable given complete information. Key quantum principles like the Heisenberg uncertainty ..."},
        {"role": "user", "content": "What did hawkins say?"},
        {"role": "assistant", "content": "It seems you are referring to Stephen Hawking, the renowned physicist. Stephen Hawking supported the principles of quantum mechanics, believing that the universe is fundamentally ..."},
        {"role": "user", "content": "So does it mean determinism is refuted?"},
        {"role": "assistant", "content": "Determinism is not entirely refuted but is significantly challenged at the quantum level, where subatomic behavior follows probabilistic principles rather than predictable, classical laws ..."},
        {"role": "user", "content": "Does amything of this have implications in a phylosofical way or is it not correct to use it in this sense?"},
        {"role": "assistant", "content": "Quantum mechanics challenges classical determinism and raises important philosophical questions in metaphysics, free will, and the nature of observation. Its probabilistic nature ..."}],
     "intent": "You are a user chatting with an assistant language model to understand how quantum mechanics challenges Eistein's determinism and get the perspective of different scientists on this."},
]


def format_conversation(conversation):
    return "\n".join(f"<{t['role']}>: {t['content']}" for t in conversation)


def intent_prompt(conversation):
    parts = ["You are given the conversation history between a user and assistant model and your task is to create a summary of the user's intent from the conversation.",
             "",
             "Your summary should be structured to define what the high level intent of the user is, but should not go into specific details.",
             "",
             'Format the summary to start with "You are a user chatting with an assistant language model to"',
             ""]
    for i, example in enumerate(FEW_SHOT, 1):
        parts += [f"Example {i}:", "", "Conversation History:", format_conversation(example["conversation"]), "",
                  "Intent Summary:", example["intent"], ""]
    parts += ["Now generate a summary of the user intent for the following conversation:", "",
              format_conversation(conversation), "", "Reply with only the intent summary and nothing else."]
    return "\n".join(parts)


class Pool:
    """Route each request to the least-loaded server."""

    def __init__(self, ports):
        self.clients = [OpenAI(api_key="EMPTY", base_url=f"http://localhost:{p}/v1") for p in ports]
        self.load = [0] * len(ports)
        self.lock = threading.Lock()

    def generate(self, prompt, model):
        with self.lock:
            i = min(range(len(self.load)), key=self.load.__getitem__)
            self.load[i] += 1
        try:
            r = self.clients[i].chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}], max_tokens=8192, temperature=0.7,
                top_p=0.8, extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content.strip()
        finally:
            with self.lock:
                self.load[i] -= 1


def generate(record, pool, model, retries=4):
    for attempt in range(retries):
        try:
            intent = pool.generate(intent_prompt(record["conversation"]), model)
            if intent:
                return {"conv_id": record["conv_id"], "conversation": record["conversation"], "intent": intent}
        except Exception as e:
            if getattr(e, "status_code", None) == 400:   # e.g. the conversation exceeds the context window
                return None
        time.sleep(2 ** attempt + random.random())
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES, required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--model", default=INTENT_MODEL)
    ap.add_argument("--ports", type=int, nargs="+", default=[8000])
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    pool = Pool(args.ports)
    for split in args.splits:
        out = USERLM_DATA / args.source / "intents" / f"{split}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(line)["conv_id"] for line in out.open()} if out.exists() else set()
        rows = [json.loads(line) for line in (CORPUS / args.source / f"{split}.jsonl").open()]
        todo = [r for r in rows if r["conv_id"] not in done]
        failed = 0
        with ThreadPoolExecutor(args.workers) as ex, out.open("a", encoding="utf-8") as f:
            for fut in tqdm(as_completed([ex.submit(generate, r, pool, args.model) for r in todo]),
                            total=len(todo), desc=f"{args.source}/{split}"):
                result = fut.result()
                if result is None:
                    failed += 1
                else:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"{args.source}/{split}: {len(todo) - failed} new intents, {failed} failed, {len(done)} existing")


if __name__ == "__main__":
    main()
