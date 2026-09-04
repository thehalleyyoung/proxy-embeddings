"""Is the decoder profile really above its encoder-to-encoder baselines?

One decoder in this study (SQL) produced a near-field profile that exceeds both
cross-family encoder baselines, and that single result is doing more work than
any other number here: it is the only case in which the shared-continuity
confound does not explain a proxy-validation correlation. A margin that size
between two correlations, computed on one corpus, deserves an interval before it
carries a claim.

Two tests:

  **Bootstrap over items.** Resample the corpus with replacement, recompute the
  decoder profile and every encoder-to-encoder baseline on the resampled items,
  and take the margin (decoder minus the largest baseline). If the interval
  includes zero, the decoder is not distinguishable from its baselines.

  **Permutation null.** Break the text-to-artifact correspondence and recompute
  the same margin. This is where the margin should go if the association is not
  real.

Bootstrapping the ITEMS rather than the pairs is deliberate: pairs are not
independent, so a pair-level resample would give an interval far too narrow.

    python3 null_test.py sql
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "code"))

import probe as P            # noqa: E402
import controls as C         # noqa: E402

B = 300


def near_minus_far(TD: np.ndarray, AD: np.ndarray, n_bins: int = 10) -> float:
    iu = P.upper_pairs(TD.shape[0])
    td, ad = TD[iu], AD[iu]
    ok = np.isfinite(td) & np.isfinite(ad)
    td, ad = td[ok], ad[ok]
    if td.size < 200:
        return np.nan
    e = np.quantile(td, np.linspace(0, 1, n_bins + 1))
    e[-1] = np.nextafter(e[-1], np.inf)
    out = []
    for b in (0, n_bins - 1):
        m = (td >= e[b]) & (td < e[b + 1])
        if m.sum() < 30 or np.ptp(td[m]) == 0 or np.ptp(ad[m]) == 0:
            return np.nan
        a, c = td[m], ad[m]
        out.append(float(((a - a.mean()) * (c - c.mean())).mean()
                         / (a.std() * c.std())))
    return out[0] - out[1]


def margin(idx, AD, views) -> float:
    """Decoder near-minus-far, less the largest encoder-to-encoder baseline."""
    sub = np.ix_(idx, idx)
    names = list(views)
    dec = near_minus_far(views[names[0]][sub], AD[sub])
    base = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            base.append(near_minus_far(views[names[i]][sub], views[names[j]][sub]))
    base = [b for b in base if np.isfinite(b)]
    if not np.isfinite(dec) or not base:
        return np.nan
    return dec - max(base)


def main() -> None:
    from pipeline import embed
    which = sys.argv[1] if len(sys.argv) > 1 else "sql"
    texts, AD = C.LOADERS[which]()
    n = len(texts)
    print(f"[{which}] {n} items")

    views = {
        "nomic": P.pairwise_cosine(embed(texts)),
        "char_tfidf": P.pairwise_cosine(C.char_tfidf(texts)),
        "word_tfidf": P.pairwise_cosine(C.word_tfidf(texts)),
    }
    names = list(views)

    obs_dec = near_minus_far(views["nomic"], AD)
    base = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            base[f"{names[i]}|{names[j]}"] = near_minus_far(
                views[names[i]], views[names[j]])
    print(f"\n  decoder near-minus-far        {obs_dec:+.3f}")
    for k, v in base.items():
        print(f"  baseline {k:<24} {v:+.3f}")
    obs = obs_dec - max(base.values())
    print(f"\n  observed margin over the LARGEST baseline   {obs:+.3f}")

    rng = np.random.default_rng(0)
    boot = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(idx)) < 30:
            continue
        m = margin(np.unique(idx), AD, views)
        if np.isfinite(m):
            boot.append(m)
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  bootstrap over items ({len(boot)} resamples)")
    print(f"    mean {boot.mean():+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"    fraction of resamples with margin > 0: {100*np.mean(boot > 0):.1f}%")

    perm = []
    for b in range(B):
        p = rng.permutation(n)
        v = near_minus_far(views["nomic"], AD[np.ix_(p, p)]) - max(base.values())
        if np.isfinite(v):
            perm.append(v)
    perm = np.array(perm)
    print(f"\n  permutation null ({len(perm)} permutations)")
    print(f"    mean {perm.mean():+.3f}  sd {perm.std():.3f}   "
          f"P(null >= observed) = {np.mean(perm >= obs):.3f}")

    out = {"decoder": which, "n_items": n, "decoder_near_minus_far": obs_dec,
           "baselines": base, "observed_margin": obs,
           "bootstrap_mean": float(boot.mean()),
           "bootstrap_ci": [float(lo), float(hi)],
           "bootstrap_frac_positive": float(np.mean(boot > 0)),
           "perm_mean": float(perm.mean()), "perm_sd": float(perm.std()),
           "perm_p": float(np.mean(perm >= obs))}
    (HERE / "runs" / f"null_{which}.json").write_text(json.dumps(out, indent=2))

    print("\n  VERDICT")
    if lo > 0:
        print("    the decoder clears its largest baseline with a CI excluding zero")
    else:
        print("    the margin's CI includes zero: the decoder is NOT distinguishable")
        print("    from its encoder-to-encoder baselines on this corpus")


if __name__ == "__main__":
    main()
