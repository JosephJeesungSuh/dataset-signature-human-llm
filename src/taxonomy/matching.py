"""Separability beyond human-designed taxonomy summaries (Sec. 2.3, App. C.2-C.3).

  features  per source/split: broad-category vectors of the three facets (with completeness) and the
            structural features (number of user turns, total cl100k_base tokens of user text)
  forward   Table 4: for each dataset pair and cumulatively matched facets (F, F+T, F+T+M, F+T+M+S),
            select maximum equally sized subsets with equal feature totals (Eq. 3) separately in train,
            validation, and test, then train and evaluate the probe on the matched subsets.
            "Original" is the probe on the full cohorts.
  reverse   Figure 4: score held-out test conversations with the Original pair classifier, select
            maximum subsets with matching score histograms (20 equal-width bins of P(second dataset)),
            and compare category distributions before and after matching. Also writes the category
            prevalences of the test cohorts (Figure 7).
  noise     Figure 4 noise floor: 1,000 random splits of each cohort into two disjoint halves of
            floor(N/2) conversations; the median category difference between halves.

The category difference of two cohorts on a facet is the mean absolute difference, in percentage points,
between the shares of conversations containing each category, using only conversations whose annotations
for that facet are complete.

    python -m src.taxonomy.matching features
    python -m src.taxonomy.matching forward
    python -m src.taxonomy.matching reverse
    python -m src.taxonomy.matching noise
"""
import argparse
import itertools
import json
import time

import numpy as np

from src.config import CORPUS, DISPLAY, EMBED_MODEL, RESULTS, SEED, SPLITS, TAXONOMY_SOURCES
from src.taxonomy.annotate import OUT as ANNOTATIONS

OUT = RESULTS / "taxonomy"
FACETS = ["function", "topic", "multiturn"]
SHORT = {"function": "F", "topic": "T", "multiturn": "M", "structure": "S"}
CONDITIONS = [FACETS[:1], FACETS[:2], FACETS, FACETS + ["structure"]]
PAIRS = list(itertools.combinations(TAXONOMY_SOURCES, 2))
BINS = 20


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def embedding_dir(source):
    from src.classification.embed import embedding_dir as d
    return d(source, EMBED_MODEL)


# ------------------------------------------------------------------------------------------- #
def build_features():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    for source in TAXONOMY_SOURCES:
        for split in SPLITS:
            rows = read_jsonl(CORPUS / source / f"{split}.jsonl")
            ids = [r["conv_id"] for r in rows]
            assert ids == [m["conv_id"] for m in read_jsonl(embedding_dir(source) / f"{split}__meta.jsonl")]
            arrays, labels = {}, {}
            for facet in FACETS:
                ann = {a["conversation_id"]: a for a in
                       read_jsonl(ANNOTATIONS / f"{source}_{split}_{facet}" / "conversations.jsonl")}
                arrays[facet] = np.asarray([ann[c]["vector"] for c in ids], dtype=np.int64)
                arrays[f"{facet}_complete"] = np.asarray([ann[c]["complete"] for c in ids], dtype=bool)
                labels[facet] = ann[ids[0]]["category_order"]
            users = [[t["content"] for t in r["conversation"] if t["role"] == "user"] for r in rows]
            arrays["structure"] = np.asarray([[len(u), sum(len(enc.encode(t, disallowed_special=())) for t in u)]
                                              for u in users], dtype=np.int64)
            (OUT / "features").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUT / "features" / f"{source}__{split}.npz", ids=np.asarray(ids), **arrays)
            (OUT / "features" / "labels.json").write_text(json.dumps(labels, indent=2))
            print(f"features {source}/{split}: {len(ids)}", flush=True)


def load_features(source, split):
    with np.load(OUT / "features" / f"{source}__{split}.npz") as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------- #
def _lp_bound_and_repair(matrix, caps, na, seconds, seed):
    """LP relaxation dual bound plus an exactly feasible integer repair; None if the bound is not attained."""
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import csc_matrix
    objective = np.r_[-np.ones(na), np.zeros(len(caps) - na)]
    scale = np.maximum(1., np.max(np.abs(matrix), axis=1) / 1000.)
    lp = linprog(objective, A_eq=csc_matrix(matrix / scale[:, None]), b_eq=np.zeros(len(matrix)),
                 bounds=np.column_stack([np.zeros(len(caps)), caps]), method="highs",
                 options={"time_limit": min(seconds, 120), "dual_feasibility_tolerance": 1e-8,
                          "primal_feasibility_tolerance": 1e-8})
    if not lp.success:
        return None
    # Any equality multipliers give a valid upper bound on the integer optimum.
    dual = np.asarray(lp.eqlin.marginals / scale, dtype=np.float64)
    reduced = objective.astype(np.longdouble) - matrix.astype(np.longdouble).T @ dual.astype(np.longdouble)
    n = int(np.floor(float(-np.minimum(reduced, 0) @ caps.astype(np.longdouble)) + 1e-4))
    if n <= 0:
        return None
    base = np.clip(np.floor(lp.x + 1e-6), 0, caps).astype(np.int64)
    valid = lambda z: int(z[:na].sum()) == n and np.all(matrix.astype(np.int64) @ z == 0)
    if valid(base):
        return base
    rng = np.random.default_rng(seed)
    fractional = np.flatnonzero(np.abs(lp.x - np.rint(lp.x)) > 1e-6)
    cheapest = np.lexsort((rng.random(len(caps)), np.abs(reduced)))
    augmented, augscale = np.vstack([matrix, -objective]), np.r_[scale, 1.]
    target = np.r_[np.zeros(len(matrix)), n]
    for width in [1000, 4000, 12000]:
        free = np.unique(np.r_[fractional, cheapest[:min(width, len(caps))],
                               rng.choice(len(caps), min(width // 2, len(caps)), replace=False)])
        fixed = base.copy()
        fixed[free] = 0
        rhs = (target - augmented @ fixed) / augscale
        fit = milp(np.zeros(len(free)), integrality=np.ones(len(free)), bounds=Bounds(0, caps[free]),
                   constraints=LinearConstraint(csc_matrix(augmented[:, free] / augscale[:, None]), rhs, rhs),
                   options={"time_limit": min(seconds, 45), "mip_rel_gap": 0.})
        if fit.x is not None:
            z = fixed.copy()
            z[free] = np.rint(fit.x).astype(np.int64)
            if np.all(z >= 0) and np.all(z <= caps) and valid(z):
                return z
        if len(free) == len(caps):
            break
    return None


def maximum_match(a, b, seconds=1800, seed=SEED):
    """Eq. (3): maximize the common size of subsets of a and b (rows = conversations, columns = features)
    subject to equal sizes and equal per-feature totals. Returns selected row indices of a and b."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    # Identical feature rows are interchangeable: optimize integer counts per distinct row.
    ua, ia, ca = np.unique(a, axis=0, return_inverse=True, return_counts=True)
    ub, ib, cb = np.unique(b, axis=0, return_inverse=True, return_counts=True)
    ia, ib = ia.ravel(), ib.ravel()
    na = len(ua)
    matrix = np.concatenate([np.vstack([np.ones(na), ua.T]), -np.vstack([np.ones(len(ub)), ub.T])], axis=1)
    caps = np.r_[ca, cb]
    z = _lp_bound_and_repair(matrix, caps, na, seconds, seed)
    if z is None:   # fall back to the exact MILP
        scale = np.maximum(1., np.max(np.abs(matrix), axis=1) / 1000.)
        fit = milp(np.r_[-np.ones(na), np.zeros(len(ub))], integrality=np.ones(len(caps)), bounds=Bounds(0, caps),
                   constraints=LinearConstraint(csc_matrix(matrix / scale[:, None]), 0, 0),
                   options={"time_limit": seconds, "mip_rel_gap": 0.})
        if fit.x is None or fit.status != 0:
            raise RuntimeError(f"no certified maximum match: {fit.message}")
        z = np.rint(fit.x).astype(np.int64)
    assert np.array_equal(ua.T @ z[:na], ub.T @ z[na:]) and z[:na].sum() == z[na:].sum()
    if z[:na].sum() == 0:
        raise RuntimeError("only the empty selection satisfies the matching constraints")
    rng = np.random.default_rng(seed)
    chosen = []
    for inverse, counts in [(ia, z[:na]), (ib, z[na:])]:
        groups = np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])
        chosen.append(np.sort(np.concatenate([rng.choice(g, int(c), replace=False) for g, c in zip(groups, counts) if c])))
    return chosen


def forward():
    from src.classification.probe import run_probe
    table = {}
    for a, b in PAIRS:
        name = f"{a}__{b}"
        best = run_probe([embedding_dir(a), embedding_dir(b)], OUT / "forward" / name / "original", names=[a, b])
        table[name] = {"Original": best["test"]["accuracy"]}
        for facets in CONDITIONS:
            cond = "+".join(SHORT[f] for f in facets)
            dest = OUT / "forward" / name / cond
            rows, sizes = {}, {}
            for split in SPLITS:
                fa, fb = load_features(a, split), load_features(b, split)
                eligible, matrices = [], []
                for f in (fa, fb):
                    mask = np.ones(len(f["ids"]), dtype=bool)
                    for facet in facets:
                        if facet != "structure":
                            mask &= f[f"{facet}_complete"]
                    ix = np.flatnonzero(mask)
                    eligible.append(ix)
                    matrices.append(np.concatenate([f[facet][ix] for facet in facets], axis=1))
                t0 = time.time()
                sa, sb = maximum_match(*matrices, seed=SEED + SPLITS.index(split))
                rows[split] = [eligible[0][sa], eligible[1][sb]]
                sizes[split] = len(sa)
                print(f"match {name} {cond} {split}: {len(sa)} per dataset ({time.time() - t0:.0f}s)", flush=True)
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / "selection.npz", **{f"{split}_{i}": r for split in SPLITS
                                                           for i, r in enumerate(rows[split])})
            best = run_probe([embedding_dir(a), embedding_dir(b)], dest, names=[a, b], rows=rows)
            table[name][cond] = best["test"]["accuracy"]
            table[name][cond + "_sizes"] = sizes
    (OUT / "forward" / "table4.json").write_text(json.dumps(table, indent=2))
    header = ["Original"] + ["+".join(SHORT[f] for f in c) for c in CONDITIONS]
    print(f"\n{'Pair':<22}" + "".join(f"{h:>10}" for h in header) + f"{'Delta':>8}")
    for name, row in table.items():
        a, b = name.split("__")
        print(f"{DISPLAY[a] + ' / ' + DISPLAY[b]:<22}" + "".join(f"{100 * row[h]:10.1f}" for h in header)
              + f"{100 * (row[header[-1]] - row['Original']):8.1f}")


# ------------------------------------------------------------------------------------------- #
def category_difference(fa, ia, fb, ib):
    """Mean absolute difference (percentage points) of category shares, per facet."""
    out = {}
    for facet in FACETS:
        shares = []
        for f, ix in ((fa, ia), (fb, ib)):
            ix = ix[f[f"{facet}_complete"][ix]]
            if not len(ix):
                raise ValueError(f"no complete {facet} annotations in a cohort")
            shares.append(f[facet][ix].mean(0))
        out[facet] = float(100 * np.abs(shares[0] - shares[1]).mean())
    return out


def score_match(scores, bins=BINS, seed=SEED):
    """Within each equal-width bin keep min(n_a, n_b) conversations of each dataset, uniformly at random."""
    edges = np.linspace(0, 1, bins + 1)
    index = [np.minimum(np.searchsorted(edges, s, side="right") - 1, bins - 1) for s in scores]
    quota = np.minimum(*[np.bincount(i, minlength=bins) for i in index])
    rng = np.random.default_rng(seed)
    return [np.sort(np.concatenate([rng.choice(np.flatnonzero(i == k), int(q), replace=False)
                                    for k, q in enumerate(quota)])) for i in index]


def pair_scores(a, b):
    """P(b) for the test conversations of a and b from the validation-selected Original classifier."""
    from scipy.special import softmax
    probe = OUT / "forward" / f"{a}__{b}" / "original"
    best = json.loads((probe / "best.json").read_text())
    with np.load(probe / f"{best['feature']}__test_logits.npz") as z:
        p = softmax(z["logits"].astype(np.float64), axis=1)[:, 1]
        lookup = dict(zip(z["conv_ids"].tolist(), p))
    return [np.asarray([lookup[c] for c in load_features(s, "test")["ids"].tolist()]) for s in (a, b)]


def prevalences():
    """Figure 7: percentage of test conversations (complete annotations only) containing each category."""
    labels = json.loads((OUT / "features" / "labels.json").read_text())
    out = {}
    for source in TAXONOMY_SOURCES:
        f = load_features(source, "test")
        out[source] = {}
        for facet in FACETS:
            share = 100 * f[facet][f[f"{facet}_complete"]].mean(0)
            out[source][facet] = dict(zip(labels[facet], share.round(1).tolist()))
    return out


def reverse():
    (OUT / "reverse").mkdir(parents=True, exist_ok=True)
    (OUT / "reverse" / "category_prevalence.json").write_text(json.dumps(prevalences(), indent=2))
    results, selections = {}, {}
    for a, b in PAIRS:
        fa, fb = load_features(a, "test"), load_features(b, "test")
        scores = pair_scores(a, b)
        sa, sb = score_match(scores)
        name = f"{a}__{b}"
        selections[f"{name}__{a}"], selections[f"{name}__{b}"] = sa, sb
        results[name] = {"n_before": [len(fa["ids"]), len(fb["ids"])], "n_after": len(sa),
                         "before": category_difference(fa, np.arange(len(fa["ids"])), fb, np.arange(len(fb["ids"]))),
                         "after": category_difference(fa, sa, fb, sb)}
    np.savez_compressed(OUT / "reverse" / "selections.npz", **selections)
    (OUT / "reverse" / "category_difference.json").write_text(json.dumps(results, indent=2))
    for name, r in results.items():
        a, b = name.split("__")
        print(f"{DISPLAY[a]} / {DISPLAY[b]} (n={r['n_after']} each): " + ", ".join(
            f"{facet} {r['before'][facet]:.2f} -> {r['after'][facet]:.2f}" for facet in FACETS))


def noise(repetitions=1000):
    """Median category difference between two random disjoint halves of the same cohort."""
    selections = np.load(OUT / "reverse" / "selections.npz")
    cohorts = {f"{s}__original": (s, np.arange(len(load_features(s, "test")["ids"]))) for s in TAXONOMY_SOURCES}
    cohorts.update({f"{a}__{b}__{s}": (s, selections[f"{a}__{b}__{s}"]) for a, b in PAIRS for s in (a, b)})
    per_rep = {}
    for ci, (cohort, (source, ix)) in enumerate(cohorts.items()):
        f = load_features(source, "test")
        reps = []
        for rep in range(repetitions):
            perm = np.random.default_rng([SEED, ci, rep]).permutation(ix)
            h = len(perm) // 2
            reps.append(category_difference(f, perm[:h], f, perm[h:2 * h]))
        per_rep[cohort] = reps
    result = {}
    for a, b in PAIRS:
        for stage in ("original", "matched"):
            keys = [f"{s}__original" for s in (a, b)] if stage == "original" else [f"{a}__{b}__{s}" for s in (a, b)]
            result[f"{a}__{b}__{stage}"] = {
                facet: float(np.median([(x[facet] + y[facet]) / 2 for x, y in zip(*(per_rep[k] for k in keys))]))
                for facet in FACETS}
    result["per_cohort"] = {k: {facet: float(np.median([r[facet] for r in v])) for facet in FACETS}
                            for k, v in per_rep.items()}
    (OUT / "reverse" / "noise_floor.json").write_text(json.dumps(result, indent=2))
    for a, b in PAIRS:
        r = result[f"{a}__{b}__original"]
        print(f"{DISPLAY[a]} / {DISPLAY[b]} noise floor: " + ", ".join(f"{k} {v:.2f}" for k, v in r.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["features", "forward", "reverse", "noise"])
    ap.add_argument("--repetitions", type=int, default=1000)
    a = ap.parse_args()
    {"features": build_features, "forward": forward, "reverse": reverse,
     "noise": lambda: noise(a.repetitions)}[a.command]()
