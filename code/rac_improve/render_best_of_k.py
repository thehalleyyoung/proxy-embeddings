"""
Best-of-K at the RENDER, not at the candidate.

§6.17 measures best-of-K on the written candidate and finds what it buys
depends on what the selector optimizes. That selection happens before the
handoff. The seam probe says the handoff is where conditioning is lost --
axes are realized in the instruction at mean rho 0.148 and in the render at
0.096 -- which puts the lossy stage downstream of every selector the pipeline
currently runs. If the loss is at the render, that is where to select.

One instruction, K renders, keep the render that most improves the packing.
This costs K image calls per item and no extra text conditioning: it spends
budget exactly where the measurement says the grip is lost.

Two selectors are run over the SAME candidates, so the comparison costs
nothing extra:

  clip     keep the render maximizing min CLIP distance to the accepted set
  pixel    keep the render maximizing min distance in optical statistics

The second is the interesting one. §6.16 reports separation on pixel channels
the method never optimizes; selecting on pixels makes that channel an objective
and asks what it costs the one the method does optimize. Reporting a gain in
the channel a selector optimizes is near-circular on its own, so the numbers
that carry the result are the CROSS-channel ones and the size of the gain
against the K^(1/m) bound of §3.

Usage:
    python3 render_best_of_k.py [run_dir] [K]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

from render_images import _load_openai_key                # noqa: E402
from axis_realization import literal_features             # noqa: E402
from probe_image_seam import render_keyed                 # noqa: E402

_load_openai_key()
K = 3


def min_nn(X: np.ndarray) -> float:
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    np.fill_diagonal(D, np.inf)
    return float(D.min())


def nn_stats(X: np.ndarray) -> dict:
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    np.fill_diagonal(D, np.inf)
    nn = D.min(1)
    return {"min_nn": float(nn.min()), "mean_nn": float(nn.mean()),
            "p5_nn": float(np.percentile(nn, 5)),
            "mean_pair": float(D[np.isfinite(D)].mean())}


def greedy(cands: list[list[int]], V: np.ndarray) -> list[int]:
    """Sequential max-min: for each item keep the candidate whose nearest
    accepted neighbour is furthest. The first item has no accepted set, so it
    takes candidate 0 -- the same render the baseline uses, which keeps the two
    arms sharing a starting point rather than diverging on an arbitrary pick."""
    chosen = [cands[0][0]]
    for row in cands[1:]:
        A = V[chosen]
        best, bd = row[0], -1.0
        for c in row:
            d = float(np.sqrt(((A - V[c]) ** 2).sum(1)).min())
            if d > bd:
                best, bd = c, d
        chosen.append(best)
    return chosen


def main(run: Path, k: int) -> None:
    log = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines() if l.strip()]
    out = HERE / "bok_renders"
    out.mkdir(parents=True, exist_ok=True)

    # candidate 0 is the run's own render; k-1 more per item
    jobs = []
    for r in log:
        for j in range(1, k):
            jobs.append((f"{int(r['i']):05d}_c{j}", r["prompt"]))
    todo = [(n, p) for n, p in jobs if not (out / f"{n}.png").exists()]
    print(f"{len(log)} items, K={k}: {len(jobs)} extra renders, {len(todo)} to make",
          flush=True)
    if todo:
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(lambda a: render_keyed(a[0], a[1], out), todo))

    paths, cands = [], []
    for r in log:
        row = []
        p0 = run / "images" / r["file"]
        if p0.exists():
            row.append(len(paths)); paths.append(p0)
        for j in range(1, k):
            p = out / f"{int(r['i']):05d}_c{j}.png"
            if p.exists():
                row.append(len(paths)); paths.append(p)
        if row:
            cands.append(row)
    full = [c for c in cands if len(c) == k]
    print(f"{len(paths)} renders on disk; {len(full)}/{len(cands)} items have all {k}",
          flush=True)
    cands = full

    from image_steer7 import Clip
    clip = Clip()
    C = clip.image(paths)
    P = np.array([[v for _, v in sorted(literal_features(p).items())] for p in paths])
    P = (P - P.mean(0)) / np.clip(P.std(0), 1e-9, None)

    base = [row[0] for row in cands]
    arms = {"baseline K=1": base,
            "best-of-K on CLIP": greedy(cands, C),
            "best-of-K on pixels": greedy(cands, P)}

    print(f"\nn = {len(base)} accepted per arm\n")
    print(f"{'arm':<22}{'CLIP min NN':>13}{'CLIP mean':>11}{'pix min NN':>12}{'pix mean':>10}")
    res = {}
    for name, sel in arms.items():
        c, p = nn_stats(C[sel]), nn_stats(P[sel])
        res[name] = {"clip": c, "pixel": p}
        print(f"{name:<22}{c['min_nn']:>13.4f}{c['mean_nn']:>11.4f}"
              f"{p['min_nn']:>12.4f}{p['mean_nn']:>10.4f}")

    b = res["baseline K=1"]
    print(f"\n{'arm':<22}{'CLIP min NN':>13}{'pix min NN':>13}   (ratio to baseline)")
    for name in ("best-of-K on CLIP", "best-of-K on pixels"):
        r_ = res[name]
        print(f"{name:<22}"
              f"{r_['clip']['min_nn'] / max(b['clip']['min_nn'], 1e-9):>13.3f}"
              f"{r_['pixel']['min_nn'] / max(b['pixel']['min_nn'], 1e-9):>13.3f}")
    # Persist embeddings and selections: min NN is a single order statistic
    # over n points and needs a bootstrap, which should not cost another 180
    # renders and a CLIP pass to run.
    np.savez(HERE / "bok_state.npz", C=C, P=P,
             **{f"sel_{i}": np.array(s_) for i, s_ in enumerate(arms.values())},
             arm_names=np.array(list(arms.keys())))
    json.dump(res, open(HERE / "render_best_of_k.json", "w"), indent=2)


if __name__ == "__main__":
    r = Path(sys.argv[1]) if len(sys.argv) > 1 else RESEARCH / "real" / "dalle_steer7_maxmin"
    main(r.resolve(), int(sys.argv[2]) if len(sys.argv) > 2 else K)
