"""Co-trained classifiers (App. E.1, Table 9): fine-tune Qwen2.5 base models end to end with a
linear classification head on WC-1M + LMSYS + ShareChat.

Inputs are the same marker-deleted, assistant-masked texts, truncated to 4,096 tokens with EOS appended.
Two epochs, effective batch size 32, cosine schedule with 5% warmup, bf16; checkpoints every 1,000 steps;
the checkpoint and the learning rate (1e-5 or 3e-5) are selected by validation balanced accuracy.

    python -m src.classification.finetune --size 0.5B
    torchrun --nproc_per_node 4 -m src.classification.finetune --size 7B --per_device_batch 2
"""
import argparse
import json
import os

import numpy as np

from src.classification.data import load_split, metrics_from_preds
from src.config import CORPUS, FOCAL, RESULTS, SEED, SPLITS

MAX_LEN = 4096
EFFECTIVE_BATCH = 32


def run(size, lr, per_device_batch, out):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments, set_seed)
    set_seed(SEED)
    model_id = f"Qwen/Qwen2.5-{size}"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        enc = tok(batch["text"], truncation=True, max_length=MAX_LEN - 1, add_special_tokens=False)
        enc["input_ids"] = [x + [tok.eos_token_id] for x in enc["input_ids"]]
        enc["attention_mask"] = [x + [1] for x in enc["attention_mask"]]
        return enc

    data = {}
    for split in SPLITS:
        texts, labels, _ = load_split(split, FOCAL, CORPUS)
        data[split] = Dataset.from_dict({"text": texts, "labels": labels}).map(
            tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=len(FOCAL), dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False

    def compute_metrics(pred):
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        m = metrics_from_preds(pred.label_ids, logits.argmax(-1), FOCAL)
        return {"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"]}

    world = int(os.environ.get("WORLD_SIZE", "1"))
    args = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=per_device_batch, per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=max(1, EFFECTIVE_BATCH // (per_device_batch * world)), learning_rate=lr,
        num_train_epochs=2, warmup_ratio=0.05, weight_decay=0.0, lr_scheduler_type="cosine", bf16=True,
        eval_strategy="steps", eval_steps=1000, save_strategy="steps", save_steps=1000, save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="balanced_accuracy", greater_is_better=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20, report_to=[], seed=SEED, train_sampling_strategy="group_by_length")
    class SequentialEvalTrainer(Trainer):
        def _get_eval_sampler(self, eval_dataset):   # length grouping applies to training batches only
            return torch.utils.data.SequentialSampler(eval_dataset)

    trainer = SequentialEvalTrainer(model=model, args=args, train_dataset=data["train"], eval_dataset=data["val"],
                                    data_collator=DataCollatorWithPadding(tok, pad_to_multiple_of=8),
                                    compute_metrics=compute_metrics)
    # The sequence-classification loss does not take num_items_in_batch; let Trainer average over accumulation.
    trainer.model_accepts_loss_kwargs = False
    trainer.train()
    result = {"model": model_id, "lr": lr, "best_checkpoint": trainer.state.best_model_checkpoint}
    for split in ("val", "test"):
        pred = trainer.predict(data[split])
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        result[split] = metrics_from_preds(pred.label_ids, np.asarray(logits).argmax(-1), FOCAL)
    if trainer.is_world_process_zero():
        (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", required=True, choices=["0.5B", "1.5B", "3B", "7B"])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 3e-5])
    ap.add_argument("--per_device_batch", type=int, default=8)
    a = ap.parse_args()
    root = RESULTS / "classification" / f"finetune_{a.size}"
    results = []
    for lr in a.lrs:
        out = root / f"lr_{lr:g}"
        out.mkdir(parents=True, exist_ok=True)
        results.append(json.loads((out / "result.json").read_text()) if (out / "result.json").exists()
                       else run(a.size, lr, a.per_device_batch, out))
    best = max(results, key=lambda r: r["val"]["balanced_accuracy"])
    (root / "best.json").write_text(json.dumps(best, indent=2))
    print(f"Qwen2.5-{a.size}: selected lr {best['lr']:g}, test accuracy {best['test']['accuracy']:.4f}")
