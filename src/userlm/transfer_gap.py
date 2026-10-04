"""Separability vs. transfer gap (Sec. 3.3, Figure 5).

  probes     binary probes for all 21 dataset pairs (frozen Qwen2.5-7B representations, probe.py recipe)
  correlate  symmetric transfer gap of each pair (Eq. 1),
                 Delta_AB = 1/2 [(NLL(A->B) - NLL(B->B)) + (NLL(B->A) - NLL(A->A))],
             from the NLL results of eval_nll.py, and its Pearson / Spearman correlation with the
             test accuracy of the A-vs-B probe, per user-model family.

    python -m src.userlm.transfer_gap probes
    python -m src.userlm.transfer_gap correlate --nll_root artifacts/results/nll
"""
import argparse
import itertools
import json
from pathlib import Path

from src.config import DISPLAY, EMBED_MODEL, RESULTS, SOURCES

PROBES = RESULTS / "probes" / "pairwise"
PAIRS = list(itertools.combinations(SOURCES, 2))


def probes():
    from src.classification.embed import embedding_dir
    from src.classification.probe import run_probe
    for a, b in PAIRS:
        out = PROBES / f"{a}__{b}"
        if not (out / "best.json").exists():
            run_probe([embedding_dir(a, EMBED_MODEL), embedding_dir(b, EMBED_MODEL)], out, names=[a, b])


def correlate(nll_root):
    from scipy.stats import pearsonr, spearmanr
    summary = {}
    for family in sorted(p.name for p in Path(nll_root).iterdir() if p.is_dir()):
        nll = {s: json.loads((Path(nll_root) / family / f"{s}.json").read_text())["per_test_set"] for s in SOURCES}
        rows = []
        for a, b in PAIRS:
            gap = 0.5 * ((nll[a][b]["nll_token_mean"] - nll[b][b]["nll_token_mean"])
                         + (nll[b][a]["nll_token_mean"] - nll[a][a]["nll_token_mean"]))
            acc = json.loads((PROBES / f"{a}__{b}" / "best.json").read_text())["test"]["accuracy"]
            rows.append({"pair": f"{DISPLAY[a]} / {DISPLAY[b]}", "accuracy": acc, "transfer_gap": gap})
        x, y = [r["accuracy"] for r in rows], [r["transfer_gap"] for r in rows]
        p, s = pearsonr(x, y), spearmanr(x, y)
        summary[family] = {"pearson_r": p.statistic, "pearson_p": p.pvalue, "spearman_rho": s.statistic,
                           "spearman_p": s.pvalue, "pairs": rows}
        print(f"{family}: Pearson r = {p.statistic:.2f} (p = {p.pvalue:.1e}), "
              f"Spearman rho = {s.statistic:.2f} (p = {s.pvalue:.1e})")
    (RESULTS / "transfer_gap.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["probes", "correlate"])
    ap.add_argument("--nll_root", type=Path, default=RESULTS / "nll",
                    help="one subdirectory per user-model family with <training source>.json from eval_nll.py")
    a = ap.parse_args()
    probes() if a.command == "probes" else correlate(a.nll_root)
