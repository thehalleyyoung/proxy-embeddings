"""Optical spread and local crowding, scored at a consistent render tier.

Corpora were rendered at different quality tiers and some hold several tiers
side by side, so a recursive glob does not select comparably across them: it
resolves `dalle_naive` to a low-quality directory and its neighbours to
high-quality ones. Every corpus here is scored from `images_high` where that
exists and from `images/` otherwise, which is the high-quality output for the
arms that keep a single directory.

Two statistics, both in the nine-dimensional pixel-statistic space and both
outside every objective in the pipeline:

  spread    mean pairwise distance -- how far apart the corpus is overall
  medNN     median nearest-neighbour distance -- how crowded it is locally,
            which mean pairwise distance cannot report and outliers cannot fake
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axis_realization import literal_features

R = Path(__file__).resolve().parent.parent / "real"
BASE = {"dalle_naive","dalle_high_temp","dalle_self_instruct",
        "dalle_evol_instruct","dalle_persona"}
N = 60

def imgs(d: Path):
    for sub in ("images_high", "images"):
        p = d / sub
        if p.is_dir():
            f = sorted(p.glob("*.png"))
            if len(f) >= N:
                return f[:N], sub
    return None, None

rows = []
for d in sorted(R.glob("dalle_*")):
    if "INVALID" in d.name: continue
    f, sub = imgs(d)
    if not f: continue
    F = np.array([[v for _, v in sorted(literal_features(p).items())] for p in f])
    rows.append((d.name, sub, F))

G = np.vstack([r[2] for r in rows]); mu, sd = G.mean(0), np.clip(G.std(0), 1e-9, None)
out = {}
for name, sub, F in rows:
    X = (F - mu) / sd
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    iu = np.triu_indices(len(X), 1)
    Dn = D.copy(); np.fill_diagonal(Dn, np.inf)
    out[name] = {"tier": sub, "spread": float(D[iu].mean()),
                 "medNN": float(np.median(Dn.min(1))),
                 "minNN": float(Dn.min(1).min())}

ours = {k: v for k, v in out.items() if k not in BASE}
base = {k: v for k, v in out.items() if k in BASE}
print(f"{'corpus':<26}{'tier':<14}{'spread':>9}{'medNN':>9}{'minNN':>9}  group")
for k, v in sorted(out.items(), key=lambda kv: -kv[1]["medNN"]):
    print(f"{k:<26}{v['tier']:<14}{v['spread']:>9.3f}{v['medNN']:>9.3f}{v['minNN']:>9.3f}"
          f"  {'baseline' if k in BASE else 'ours'}")
from scipy.stats import mannwhitneyu
for stat in ("spread", "medNN", "minNN"):
    o = [v[stat] for v in ours.values()]; b = [v[stat] for v in base.values()]
    inv = sum(1 for x in o for y in b if x <= y)
    u = mannwhitneyu(o, b, alternative="greater")
    print(f"\n{stat}: ours {min(o):.3f}-{max(o):.3f}  base {min(b):.3f}-{max(b):.3f}  "
          f"ratio {np.mean(o)/np.mean(b):.3f}x  inversions {inv}/{len(o)*len(b)}  "
          f"U={u.statistic:.0f} p={u.pvalue:.2g}")
json.dump(out, open(Path(__file__).parent / "optical_consistent.json", "w"), indent=2)
