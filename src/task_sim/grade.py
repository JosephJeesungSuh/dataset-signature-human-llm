"""Grade simulated task conversations and summarize task success (Sec. 3.2, Table 6).

Every assistant turn is verified; a conversation succeeds if any assistant turn solves the problem.
  GSM8K      an LLM verifier (gpt-5.6-luna, low reasoning effort) compares the response's final answer
             with the reference answer of the original problem
  HumanEval  the code in the response is run against the problem's unit tests (subprocess, 10 s timeout)
Table 6 reports success rates averaged over 100 problems x 5 conversations, with a 90% Student's t interval
over the problem-level averages; Delta (Qwen minus Llama) uses the paired problem-level differences.

    OPENAI_API_KEY=... python -m src.task_sim.grade grade
    python -m src.task_sim.grade summarize
"""
import argparse
import ast
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.config import DISPLAY, RESULTS, SOURCES
from src.task_sim.simulate import ASSISTANT_MODELS, OUT as CONVERSATIONS, assistant_tag
from src.task_sim.tasks import DATASETS, TASKS

OUT = RESULTS / "task_sim"
JUDGE_MODEL, JUDGE_EFFORT = "gpt-5.6-luna", "low"
JUDGE_INSTRUCTIONS = """You grade whether an AI assistant's response solves a math word problem.

The response is one assistant turn from a multi-turn conversation with a user who wanted this problem solved. The user may have stated the problem piecemeal, imprecisely, or with changes. Grade the response only against the original problem and its reference answer, not against what the user may have said.

Set correct to true only if the response commits to a final numerical answer to the original problem that equals the reference answer. Ignore formatting, units, currency symbols, thousands separators, and equivalent forms of the same quantity. Set correct to false if the response gives no final answer (for example, it only asks a question or outlines an approach), gives a different number, gives several conflicting final answers, or answers a different question.

In final_answer, quote the response's final answer, or "none". In explanation, justify the verdict in one or two sentences."""
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False,
                "properties": {"final_answer": {"type": "string"}, "explanation": {"type": "string"},
                               "correct": {"type": "boolean"}},
                "required": ["final_answer", "explanation", "correct"]}


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def assistant_turns(record):
    return [t["content"] for t in record["conversation"] if t["role"] == "assistant"]


# ------------------------------------------------------------------------------------------- #
async def judge(items, path, concurrency=32):
    import openai
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=600, max_retries=0)
    sem = asyncio.Semaphore(concurrency)

    async def one(item, f):
        async with sem:
            gold = item["task"]["gold"]
            prompt = (f"## Original problem\n{item['task']['problem']}\n\n## Reference solution\n{gold['solution']}\n\n"
                      f"## Reference answer\n{gold['answer']}\n\n## Assistant response\n{item['text']}")
            record = {"conv_id": item["conv_id"], "turn": item["turn"], "status": "error"}
            for attempt in range(5):
                try:
                    r = await client.responses.create(
                        model=JUDGE_MODEL, reasoning={"effort": JUDGE_EFFORT}, instructions=JUDGE_INSTRUCTIONS,
                        input=prompt, max_output_tokens=16384, store=False,
                        text={"format": {"type": "json_schema", "name": "grade", "strict": True,
                                         "schema": JUDGE_SCHEMA}})
                    if r.status == "completed" and r.output_text:
                        record.update(status="ok", correct=bool(json.loads(r.output_text)["correct"]))
                        break
                except openai.APIStatusError as exc:
                    if exc.status_code in {400, 401, 403, 404}:
                        raise
                except Exception:
                    pass
                await asyncio.sleep(min(2 ** (attempt + 1), 60))
            f.write(json.dumps(record) + "\n")
            f.flush()

    try:
        with open(path, "a") as f:
            await asyncio.gather(*(one(item, f) for item in items))
    finally:
        await client.close()


# ------------------------------------------------------------------------------------------- #
_KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


def extract_code(response):
    """Imports, definitions, and assignments from the Python code blocks (or the whole reply)."""
    import re
    blocks = [code for lang, code in re.findall(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)```", response, re.DOTALL)
              if lang.lower() in ("", "python", "py", "python3")] or [response]
    nodes = []
    for code in blocks:
        try:
            nodes.extend(n for n in ast.parse(code).body if isinstance(n, _KEEP))
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            continue
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes):
        return None
    try:
        return ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    except RecursionError:
        return None


# The problem prompt runs first so that helper functions it defines exist; the candidate then replaces the
# entry point. If the entry point is missing but exactly one new function is defined, that one is tested.
HARNESS = r'''
import builtins, faulthandler, json, os, resource, shutil, subprocess, sys
spec = json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
faulthandler.disable()
exit_now = os._exit
# Keep candidate code away from the file system and processes (as human-eval's reliability_guard).
for module, names in ((os, ["kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork",
                            "forkpty", "killpg", "rename", "renames", "truncate", "replace", "unlink", "fchmod",
                            "fchown", "chmod", "chown", "chroot", "lchown", "chdir"]),
                      (shutil, ["rmtree", "move", "chown"]), (subprocess, ["Popen"])):
    for name in names:
        if hasattr(module, name):
            setattr(module, name, None)
builtins.exit = builtins.quit = None
def done(passed):
    sys.stdout.write("\n__RESULT__" + json.dumps(passed) + "\n"); sys.stdout.flush(); exit_now(0)
ns = {"__name__": "__candidate__"}
try:
    exec(compile(spec["prompt"], "prompt", "exec"), ns)
    stub = ns.get(spec["entry_point"])
    exec(compile(spec["code"], "candidate", "exec"), ns)
    fn = ns.get(spec["entry_point"])
    if fn is stub:
        if not spec["alias"]:
            done(False)
        fn = ns[spec["alias"]]
    exec(compile(spec["test"], "test", "exec"), ns)
    ns["check"](fn)
except BaseException:
    done(False)
done(True)
'''


def unit_test(task, response, timeout=10.0):
    gold = task["gold"]
    code = extract_code(response)
    if code is None:
        return False
    functions = lambda src: [n.name for n in ast.parse(src).body
                             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    alias = None
    if gold["entry_point"] not in functions(code):
        new = [n for n in functions(code) if n not in functions(gold["prompt"])]
        if len(new) != 1:
            return False
        alias = new[0]
    spec = {"prompt": gold["prompt"], "code": code, "test": gold["test"], "entry_point": gold["entry_point"],
            "alias": alias}
    with tempfile.TemporaryDirectory() as cwd:
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", HARNESS], input=json.dumps(spec), cwd=cwd,
                                  capture_output=True, text=True, timeout=timeout, start_new_session=True)
        except subprocess.TimeoutExpired:
            return False
    lines = [line for line in proc.stdout.splitlines() if line.startswith("__RESULT__")]
    return bool(lines) and json.loads(lines[-1][len("__RESULT__"):])


# ------------------------------------------------------------------------------------------- #
def grade_all():
    tasks = {d: {t["task_id"]: t for t in read_jsonl(TASKS / f"{d}.jsonl")} for d in DATASETS}
    for model in ASSISTANT_MODELS:
        for user in SOURCES:
            for dataset in DATASETS:
                convs = read_jsonl(CONVERSATIONS / assistant_tag(model) / user / f"{dataset}.jsonl")
                path = OUT / "grades" / assistant_tag(model) / user / f"{dataset}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                done = {(g["conv_id"], g["turn"]) for g in read_jsonl(path) if g["status"] == "ok"}
                items = [{"conv_id": c["conv_id"], "turn": i, "text": text, "task": tasks[dataset][c["task_id"]]}
                         for c in convs for i, text in enumerate(assistant_turns(c), 1)
                         if (c["conv_id"], i) not in done]
                if not items:
                    continue
                print(f"{assistant_tag(model)}/{user}/{dataset}: grading {len(items)} assistant turns", flush=True)
                if dataset == "gsm8k":
                    asyncio.run(judge(items, path))
                else:
                    with ThreadPoolExecutor(16) as ex, open(path, "a") as f:
                        for item, passed in zip(items, ex.map(lambda it: unit_test(it["task"], it["text"]), items)):
                            f.write(json.dumps({"conv_id": item["conv_id"], "turn": item["turn"], "status": "ok",
                                                "correct": passed}) + "\n")


def problem_success(model, user, dataset):
    """Per problem, the mean over its conversations of any-turn success."""
    grades = {(g["conv_id"], g["turn"]): g["correct"] for g in
              read_jsonl(OUT / "grades" / assistant_tag(model) / user / f"{dataset}.jsonl") if g["status"] == "ok"}
    by_task = defaultdict(list)
    for c in read_jsonl(CONVERSATIONS / assistant_tag(model) / user / f"{dataset}.jsonl"):
        turns = [(c["conv_id"], i) for i in range(1, len(assistant_turns(c)) + 1)]
        if not all(t in grades for t in turns):
            raise RuntimeError(f"ungraded turns in {model}/{user}/{dataset}; run the grade command again")
        by_task[c["task_id"]].append(any(grades[t] for t in turns))
    return {task: float(np.mean(v)) for task, v in by_task.items()}


def interval(values, level=0.90):
    from scipy.stats import t
    x = 100 * np.asarray(values)
    half = t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {"mean": float(x.mean()), "ci_half_width": float(half), "n_problems": len(x)}


def summarize():
    qwen, llama = ASSISTANT_MODELS
    table = {}
    for user in SOURCES:
        for dataset in DATASETS:
            q, l = problem_success(qwen, user, dataset), problem_success(llama, user, dataset)
            ids = sorted(q)
            assert set(ids) == set(l)
            table[f"{user}/{dataset}"] = {
                "qwen": interval([q[i] for i in ids]), "llama": interval([l[i] for i in ids]),
                "delta": interval([q[i] - l[i] for i in ids])}
    (OUT / "table6.json").write_text(json.dumps(table, indent=2))
    fmt = lambda r: f"{r['mean']:5.1f}±{r['ci_half_width']:.1f}"
    print(f"{'User model':<11}" + "".join(f"{d + ' ' + k:>17}" for d in DATASETS for k in ("Qwen", "Llama", "Delta")))
    for user in SOURCES:
        print(f"{DISPLAY[user]:<11}" + "".join(f"{fmt(table[f'{user}/{d}'][k]):>17}"
                                               for d in DATASETS for k in ("qwen", "llama", "delta")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["grade", "summarize"])
    grade_all() if ap.parse_args().command == "grade" else summarize()
