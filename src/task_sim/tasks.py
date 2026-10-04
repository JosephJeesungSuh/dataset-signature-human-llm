"""Select the GSM8K and HumanEval problems for the assistant evaluation (Sec. 3.2).

100 problems per dataset (seed 0): problems sharded by Lost in Conversation (Laban et al.) come first,
the rest are drawn at random. Each problem becomes the user model's intent:
    "You are a user chatting with an assistant language model to complete the following: <problem>"

    python -m src.task_sim.tasks
Output: artifacts/data/tasks/{gsm8k,humaneval}.jsonl
"""
import argparse
import json
import random
import re

from src.config import DATA

TASKS = DATA / "tasks"
DATASETS = ("gsm8k", "humaneval")
INTENT_TEMPLATE = "You are a user chatting with an assistant language model to complete the following: {problem}"
LIC_REPO, LIC_FILE = "microsoft/lost_in_conversation", "lost_in_conversation.json"
REVISIONS = {"openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
             "openai/openai_humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
             LIC_REPO: "b6b71f418423c61cef0f439c610a4a655cc0e2d5"}


def lic_task_ids():
    from huggingface_hub import hf_hub_download
    ids = set()
    path = hf_hub_download(LIC_REPO, LIC_FILE, repo_type="dataset", revision=REVISIONS[LIC_REPO])
    for rec in json.loads(open(path).read()):
        src, _, num = rec["task_id"].partition("/")
        if src == "sharded-GSM8K":
            ids.add(f"gsm8k/{num}")
        elif src == "sharded-HumanEval":
            ids.add(f"HumanEval/{num}")
    return ids


def load_pool(dataset):
    from datasets import load_dataset
    if dataset == "gsm8k":
        tasks = []
        rows = load_dataset("openai/gsm8k", "main", split="test", revision=REVISIONS["openai/gsm8k"])
        for i, row in enumerate(rows):
            answer = re.search(r"####\s*(.+?)\s*$", row["answer"].strip()).group(1).replace(",", "").strip()
            tasks.append({"task_id": f"gsm8k/{i}", "dataset": dataset, "problem": row["question"].strip(),
                          "gold": {"answer": answer, "solution": row["answer"]}})
        return tasks
    rows = load_dataset("openai/openai_humaneval", split="test", revision=REVISIONS["openai/openai_humaneval"])
    return [{"task_id": row["task_id"], "dataset": dataset, "problem": row["prompt"].strip("\n"),
             "gold": {"prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"],
                      "canonical_solution": row["canonical_solution"]}} for row in rows]


def select(pool, lic, n, seed):
    rng = random.Random(seed)
    first = [t for t in pool if t["task_id"] in lic]
    rest = [t for t in pool if t["task_id"] not in lic]
    chosen = rng.sample(first, n) if n < len(first) else first + rng.sample(rest, n - len(first))
    order = {t["task_id"]: i for i, t in enumerate(pool)}
    return sorted(chosen, key=lambda t: order[t["task_id"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    lic = lic_task_ids()
    TASKS.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        chosen = select(load_pool(dataset), lic, a.n, a.seed)
        with (TASKS / f"{dataset}.jsonl").open("w") as f:
            for t in chosen:
                f.write(json.dumps({**t, "intent": INTENT_TEMPLATE.format(problem=t["problem"])}) + "\n")
        print(f"{dataset}: {len(chosen)} problems")
