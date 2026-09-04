"""Internal measurement primitives for decodergap.

The public surface is in `decodergap/__init__.py`; nothing here is stable API.

Original module docstring follows.

---

decodergap — is a text-space decision safe for your decoder?

A synthetic-data pipeline embeds text and then makes decisions with the
distances: drop near-duplicates, pick a diverse subset, retrieve neighbours,
flag contamination. When the text is later *decoded* into the artifact that
actually matters — a program that runs, an SVG that renders, a query that
returns rows — those decisions are being made in a space that is not the
artifact's.

This module measures whether that substitution is safe, separately for each
kind of decision, because the answer differs by decision and a single global
correlation hides it.

The central measurement is the **near-field profile**: the correlation between
text distance and artifact distance computed *within* each decile of text
distance, rather than pooled. Pooled correlation is dominated by the near field
and reads as reassuring even when the far field carries no signal at all.

    profile = near_field_profile(text_emb, art_dist)
    verdicts = decision_verdicts(text_emb, art_dist, coverage_fn)

Nothing here is specific to a domain: supply an embedding matrix and a pairwise
artifact-distance matrix and every number follows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import stats


# ---------------------------------------------------------------- distances

def pairwise_cosine(E: np.ndarray) -> np.ndarray:
    """Cosine distance between unit-normalized rows."""
    E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    return np.clip(1.0 - E @ E.T, 0.0, 2.0)


def upper_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


# ------------------------------------------------------------ near field

@dataclass
class Decile:
    decile: int
    lo: float
    hi: float
    n_pairs: int
    corr: float
    p: float
    mean_artifact_distance: float
    frac_artifact_identical: float


@dataclass
class Profile:
    n_items: int
    n_pairs: int
    global_corr: float
    global_p: float
    near_corr: float          # closest decile
    far_corr: float           # farthest decile
    near_minus_far: float
    deciles: list[Decile] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)


def near_field_profile(text_emb: np.ndarray | None,
                       art_dist: np.ndarray,
                       text_dist: np.ndarray | None = None,
                       n_bins: int = 10,
                       identical_tol: float = 1e-9) -> Profile:
    """Correlation between text distance and artifact distance, by decile.

    `text_emb` is embedded and cosine-distanced, or pass `text_dist` directly.
    `art_dist` is any pairwise distance in the artifact's own space.

    The pooled correlation is reported beside the per-decile ones because the
    gap between them is the finding: a large pooled correlation with a flat far
    field means the representation separates near-duplicates from everything
    else and says nothing beyond that.
    """
    if text_dist is None:
        if text_emb is None:
            raise ValueError("supply text_emb or text_dist")
        text_dist = pairwise_cosine(text_emb)
    n = text_dist.shape[0]
    iu = upper_pairs(n)
    td, ad = text_dist[iu], art_dist[iu]
    ok = np.isfinite(td) & np.isfinite(ad)
    td, ad = td[ok], ad[ok]

    gr, gp = stats.pearsonr(td, ad) if td.size > 2 else (np.nan, np.nan)

    edges = np.quantile(td, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    rows: list[Decile] = []
    for b in range(n_bins):
        m = (td >= edges[b]) & (td < edges[b + 1])
        if m.sum() < 3 or np.ptp(td[m]) == 0 or np.ptp(ad[m]) == 0:
            r, p = np.nan, np.nan
        else:
            r, p = stats.pearsonr(td[m], ad[m])
        rows.append(Decile(
            decile=b + 1, lo=float(edges[b]), hi=float(edges[b + 1]),
            n_pairs=int(m.sum()), corr=float(r), p=float(p),
            mean_artifact_distance=float(ad[m].mean()) if m.any() else np.nan,
            frac_artifact_identical=float((ad[m] <= identical_tol).mean()) if m.any() else np.nan))

    return Profile(n_items=n, n_pairs=int(td.size),
                   global_corr=float(gr), global_p=float(gp),
                   near_corr=rows[0].corr, far_corr=rows[-1].corr,
                   near_minus_far=float(rows[0].corr - rows[-1].corr),
                   deciles=rows)


# --------------------------------------------------------- decision rules

def greedy_maxmin(text_dist: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """Farthest-point traversal in text space (Gonzalez), the packing selector."""
    n = text_dist.shape[0]
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    picks = [first]
    d = text_dist[first].copy()
    while len(picks) < min(k, n):
        nxt = int(np.argmax(d))
        picks.append(nxt)
        d = np.minimum(d, text_dist[nxt])
    return picks


def filter_then_sample(text_dist: np.ndarray, k: int, t: float,
                       seed: int = 0) -> list[int]:
    """Drop near-duplicates at radius t, then sample uniformly from survivors.

    Uses only the near field — the half of the distance distribution that
    carries signal — and takes no position on which of two far-apart items is
    farther apart, which is the judgement the far field cannot support.
    """
    n = text_dist.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    kept: list[int] = []
    for i in order:
        if all(text_dist[i, j] > t for j in kept):
            kept.append(int(i))
    if len(kept) >= k:
        return list(rng.choice(kept, size=k, replace=False))
    rest = [i for i in order if i not in set(kept)]
    return kept + rest[:k - len(kept)]


def random_pick(n: int, k: int, seed: int = 0) -> list[int]:
    return list(np.random.default_rng(seed).choice(n, size=min(k, n), replace=False))


def greedy_oracle(coverage_fn, n: int, k: int, cand: int = 0,
                  seed: int = 0) -> list[int]:
    """Greedy marginal coverage in the ARTIFACT's own space.

    This is the selector a practitioner is assumed not to be able to afford.
    For a deterministic decoder that runs in milliseconds it is usually cheaper
    than embedding the corpus, which is the point of measuring it here.
    """
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    remaining = list(range(n))
    cur = 0.0
    while len(picks) < min(k, n):
        pool = remaining if cand <= 0 or len(remaining) <= cand else list(
            rng.choice(remaining, size=cand, replace=False))
        best, best_gain = pool[0], -np.inf
        for i in pool:
            g = coverage_fn(picks + [i]) - cur
            if g > best_gain:
                best, best_gain = i, g
        picks.append(best)
        cur += best_gain
        remaining.remove(best)
    return picks


def selector_comparison(text_dist: np.ndarray,
                        coverage_fn,
                        budgets: list[int],
                        n_seeds: int = 20,
                        thresholds: tuple[float, ...] = (),
                        oracle: bool = True,
                        oracle_cand: int = 120) -> dict:
    """Artifact-space coverage bought by each text-space selector.

    `coverage_fn(indices) -> float` is scored in the artifact's own space, so
    no selector can win by agreeing with the representation it selected in.
    """
    n = text_dist.shape[0]
    iu = upper_pairs(n)
    td = text_dist[iu]
    ts = thresholds or tuple(np.quantile(td, [0.05, 0.10, 0.20, 0.30, 0.40]))

    out: dict = {"budgets": budgets, "n_seeds": n_seeds,
                 "thresholds": [float(t) for t in ts], "arms": {}}
    for k in budgets:
        rnd = [coverage_fn(random_pick(n, k, s)) for s in range(n_seeds)]
        mm = [coverage_fn(greedy_maxmin(text_dist, k, s)) for s in range(n_seeds)]
        out["arms"].setdefault("random", {})[k] = _stat(rnd)
        out["arms"].setdefault("maxmin", {})[k] = _stat(mm)
        for t in ts:
            fs = [coverage_fn(filter_then_sample(text_dist, k, float(t), s))
                  for s in range(n_seeds)]
            out["arms"].setdefault(f"filter@{t:.4f}", {})[k] = _stat(fs)
        if oracle:
            orc = [coverage_fn(greedy_oracle(coverage_fn, n, k, oracle_cand, s))
                   for s in range(min(5, n_seeds))]
            out["arms"].setdefault("oracle", {})[k] = _stat(orc)
    return out


def _stat(v: list[float]) -> dict:
    a = np.asarray(v, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "n": int(a.size)}


def dedup_quality(text_dist: np.ndarray, art_dist: np.ndarray,
                  identical_tol: float = 1e-9,
                  n_thresholds: int = 40) -> dict:
    """Can a text-space radius find artifact-space duplicates?

    Scores the near-field decision directly: treating "artifact distance is
    zero" as the label, sweep the text-space radius and report the best
    achievable F1 with its precision and recall, plus the AUC of the ranking.
    """
    n = text_dist.shape[0]
    iu = upper_pairs(n)
    td, ad = text_dist[iu], art_dist[iu]
    y = (ad <= identical_tol)
    if y.sum() == 0 or y.all():
        return {"n_duplicate_pairs": int(y.sum()), "auc": None, "best": None}
    # AUC that a small text distance ranks duplicate pairs first
    order = np.argsort(td)
    yr = y[order]
    ranks = np.arange(1, yr.size + 1)
    pos, neg = yr.sum(), (~yr).sum()
    auc = float((ranks[yr].sum() - pos * (pos + 1) / 2) / (pos * neg))
    auc = 1.0 - auc  # small distance should mean duplicate

    best = None
    for t in np.quantile(td, np.linspace(0.001, 0.5, n_thresholds)):
        pred = td <= t
        tp = float((pred & y).sum())
        if tp == 0:
            continue
        prec, rec = tp / pred.sum(), tp / y.sum()
        f1 = 2 * prec * rec / (prec + rec)
        if best is None or f1 > best["f1"]:
            best = {"t": float(t), "precision": prec, "recall": rec, "f1": f1}
    return {"n_duplicate_pairs": int(y.sum()),
            "duplicate_pair_rate": float(y.mean()), "auc": auc, "best": best}


def reject_purity(text_dist: np.ndarray, art_dist: np.ndarray,
                  quantiles: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40),
                  far_tol: float = 0.5) -> dict:
    """What a near-duplicate filter actually throws away, pair by pair.

    A decile-level correlation licenses a statement about the corpus. It does
    not license a per-item accept/reject, and the difference is measurable:
    among the pairs a filter at radius t would call duplicates, what fraction
    are in fact behaviourally far apart? A near-field correlation can be real
    and this rate can still be most of the rejects, because a correlation of
    0.2 over many pairs is a weak signal on any single one.
    """
    n = text_dist.shape[0]
    iu = upper_pairs(n)
    td, ad = text_dist[iu], art_dist[iu]
    pool_mean = float(ad.mean())
    rows = []
    for q in quantiles:
        t_q = float(np.quantile(td, q))
        m = td <= t_q
        if m.sum() == 0:
            continue
        rows.append({"quantile": q, "t": t_q, "n_rejected": int(m.sum()),
                     "frac_still_far": float((ad[m] > far_tol).mean()),
                     "mean_artifact_distance": float(ad[m].mean())})
    return {"far_tol": far_tol, "pool_mean_artifact_distance": pool_mean,
            "rows": rows}


# ------------------------------------------------------------- the verdict

def decision_verdicts(profile: Profile, dedup: dict, selectors: dict | None,
                      purity: dict | None = None, near_p: float = 0.05) -> dict:
    """Per-decision-rule verdicts, in the form a practitioner acts on."""
    v: dict = {}
    near_ok = (np.isfinite(profile.near_corr) and profile.near_corr > 0
               and profile.deciles[0].p < near_p)
    far_ok = (np.isfinite(profile.far_corr) and profile.far_corr > 0
              and profile.deciles[-1].p < near_p)

    # Two verdicts, because the evidence for them is different. A decile-level
    # correlation is evidence about the corpus; acting on a single pair needs
    # the pair-level error rate, and the two can disagree sharply.
    v["deduplication_aggregate"] = {
        "scope": "corpus-scale statements: contamination screening, duplicate rate",
        "verdict": "SUPPORTED" if near_ok else "UNSUPPORTED",
        "near_field_corr": profile.near_corr,
        "duplicate_detection_auc": dedup.get("auc"),
        "why": ("text distance tracks artifact distance among the closest pairs, "
                "which is the regime this decision reads")
        if near_ok else "no measurable near-field signal in this decoder"}

    pair = {"scope": "per-item accept/reject: dropping one member of a pair",
            "best_radius": (dedup.get("best") or {}).get("t"),
            "precision_on_artifact_identical": (dedup.get("best") or {}).get("precision"),
            "recall_on_artifact_identical": (dedup.get("best") or {}).get("recall")}
    if purity and purity["rows"]:
        tight = purity["rows"][0]
        pair["frac_rejected_pairs_still_far"] = tight["frac_still_far"]
        pair["at_radius"] = tight["t"]
        pair["verdict"] = ("SUPPORTED" if tight["frac_still_far"] < 0.10
                           else "NOT SUPPORTED")
        pair["why"] = (f"at the tightest radius tested, "
                       f"{100*tight['frac_still_far']:.0f}% of the pairs this "
                       f"filter would call duplicates are in fact far apart in "
                       f"the artifact (>{purity['far_tol']}), against a "
                       f"pool-wide mean of "
                       f"{purity['pool_mean_artifact_distance']:.3f}")
    else:
        pair["verdict"] = "NOT MEASURED"
    v["deduplication_per_item"] = pair

    v["max_min_selection"] = {
        "verdict": "SAFE" if far_ok else "UNSAFE",
        "far_field_corr": profile.far_corr,
        "why": ("farthest-point traversal draws exclusively from the top decile "
                "of text distance, where this decoder shows no relation between "
                "text distance and artifact distance")
        if not far_ok else "far-field signal present"}

    if selectors:
        ks = selectors["budgets"]
        rnd = [selectors["arms"]["random"][k]["mean"] for k in ks]
        mm = [selectors["arms"]["maxmin"][k]["mean"] for k in ks]
        v["max_min_selection"]["coverage_vs_random"] = {
            str(k): {"maxmin": m, "random": r, "delta": m - r}
            for k, m, r in zip(ks, mm, rnd)}
        v["max_min_selection"]["loses_to_random_at"] = [
            str(k) for k, m, r in zip(ks, mm, rnd) if m < r]
        # The verdict that matters is not maxmin-vs-random but either-vs-oracle:
        # a text-space selector can beat random and still leave most of the
        # decoder's reachable behaviour unbought.
        if "oracle" in selectors["arms"]:
            orc = [selectors["arms"]["oracle"][k]["mean"] for k in ks]
            v["max_min_selection"]["frac_of_oracle"] = {
                str(k): {"maxmin": m / o if o else None,
                         "random": r / o if o else None}
                for k, m, r, o in zip(ks, mm, rnd, orc)}

        best_arm, best_gain = None, -np.inf
        for arm, per_k in selectors["arms"].items():
            if not arm.startswith("filter@"):
                continue
            g = float(np.mean([per_k[k]["mean"] - selectors["arms"]["random"][k]["mean"]
                               for k in ks]))
            if g > best_gain:
                best_arm, best_gain = arm, g
        v["filter_then_sample"] = {
            "best_arm": best_arm, "mean_gain_over_random": best_gain,
            "verdict": "RECOMMENDED" if best_gain > 0 else "NO BENEFIT"}
    return v


def report(profile: Profile, dedup: dict, selectors: dict | None,
           domain: str, purity: dict | None = None) -> dict:
    return {"domain": domain,
            "profile": asdict(profile),
            "dedup": dedup,
            "reject_purity": purity,
            "selectors": selectors,
            "verdicts": decision_verdicts(profile, dedup, selectors, purity)}


def print_profile(p: Profile, name: str = "") -> None:
    print(f"\n  near-field profile {name}  ({p.n_items} items, {p.n_pairs} pairs)")
    print(f"  pooled corr {p.global_corr:+.3f}  (p={p.global_p:.1e})")
    print("  decile  text-dist range        n      corr      p     mean art.dist  %identical")
    for d in p.deciles:
        print(f"   {d.decile:>3}    [{d.lo:.3f}, {d.hi:.3f}]  {d.n_pairs:>6}  "
              f"{d.corr:+.3f}  {d.p:>8.2g}   {d.mean_artifact_distance:.3f}    "
              f"{100*d.frac_artifact_identical:5.1f}%")
    print(f"  near {p.near_corr:+.3f}   far {p.far_corr:+.3f}   "
          f"near-far {p.near_minus_far:+.3f}")
