"""Classifier-score KL reduction vs. downstream NLL improvement (App. E.5, Eq. 4, Figure 9).

For each target, budget, and user-model family:
    Delta_KL  = KL(p_T || p_R) - KL(p_T || p_G)
    Delta_NLL = NLL_R - NLL_G
p_T, p_R, p_G are distributions of the target classifier's scores for the target's test conversations, the
randomly selected donors, and the classifier-selected donors. Scores are binned into 20 bins at the quantiles
of the pooled training scores (target and donor pools weighted equally), with a pseudocount of 0.5 per bin.
Conversations longer than the 4,096-token SFT limit for the family are excluded, as in training.
NLL_R and NLL_G are the target-test NLLs of user models trained on the two selections, read from
<nll_root>/<family>/<target>__<method>_<pct>.json (eval_nll.py with --test_sources <target>).

    python -m src.datamix.kl_reduction --nll_root artifacts/results/nll_datamix
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.config import CORPUS, RESULTS, SOURCES, USER_BASE_MODELS, USERLM_DATA, model_tag
from src.datamix.select import BUDGET, PERCENTAGES, sft_ids

BINS, PSEUDOCOUNT, MAX_LEN = 20, 0.5, 4096
FAMILIES = {"qwen": USER_BASE_MODELS[0], "llama": USER_BASE_MODELS[1]}


def sample_lengths(source, base, split):
    with open(USERLM_DATA / source / "sft" / f"{split}_{model_tag(base)}_samples.jsonl") as f:
        return np.asarray([len(json.loads(line)["input_ids"]) for line in f])


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


def histogram(scores, edges):
    counts = np.histogram(scores, bins=edges)[0] + PSEUDOCOUNT
    return counts / counts.sum()


def main():
    from scipy.stats import pearsonr
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nll_root", type=Path, default=RESULTS / "nll_datamix")
    nll_root = ap.parse_args().nll_root
    rows = []
    for target in SOURCES:
        out = RESULTS / "datamix" / target
        donors = [s for s in SOURCES if s != target]
        scores, orders = np.load(out / "scores.npz"), np.load(out / "selection_orders.npz")
        donor_scores = np.concatenate([scores[f"{s}__train"] for s in donors])
        target_train = scores[f"{target}__train"]
        # Bin edges: quantiles of the training scores with equal total weight on target and donors.
        x = np.concatenate([target_train, donor_scores])
        w = np.concatenate([np.full(len(target_train), .5 / len(target_train)),
                            np.full(len(donor_scores), .5 / len(donor_scores))])
        order = np.argsort(x)
        inner = np.interp(np.arange(1, BINS) / BINS, np.cumsum(w[order]), x[order])
        edges = np.r_[-np.inf, np.unique(inner), np.inf]
        for family, base in FAMILIES.items():
            donor_len = np.concatenate([sample_lengths(s, base, "train") for s in donors])
            # Target-test scores follow the corpus order; keep conversations with an SFT sample within the limit.
            test_ids = sft_ids(target, base, "test")
            order_ids = [json.loads(line)["conv_id"] for line in open(CORPUS / target / "test.jsonl")]
            position = {c: i for i, c in enumerate(order_ids)}
            keep = [position[c] for c, n in zip(test_ids, sample_lengths(target, base, "test")) if n <= MAX_LEN]
            p_t = histogram(scores[f"{target}__test"][keep], edges)
            for pct in PERCENTAGES:
                sel = {m: orders[m][:BUDGET * pct // 100] for m in ("dsir", "random")}
                sel = {m: ix[donor_len[ix] <= MAX_LEN] for m, ix in sel.items()}
                p_g, p_r = histogram(donor_scores[sel["dsir"]], edges), histogram(donor_scores[sel["random"]], edges)
                nll = {m: json.loads((nll_root / family / f"{target}__{m}_{pct}.json").read_text())
                       ["per_test_set"][target]["nll_token_mean"] for m in ("dsir", "random")}
                rows.append({"target": target, "family": family, "percentage": pct,
                             "delta_kl": kl(p_t, p_r) - kl(p_t, p_g), "delta_nll": nll["random"] - nll["dsir"]})
    summary = {"rows": rows}
    for family in FAMILIES:
        sub = [r for r in rows if r["family"] == family]
        r = pearsonr([s["delta_kl"] for s in sub], [s["delta_nll"] for s in sub])
        summary[family] = {"pearson_r": r.statistic, "p_value": r.pvalue, "n": len(sub)}
        print(f"{family}: Pearson r = {r.statistic:.2f} over {len(sub)} target x budget points")
    (RESULTS / "datamix" / "kl_reduction.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
