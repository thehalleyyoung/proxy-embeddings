"""How much of the decode budget do you actually need?

The oracle — select greedily in the artifact's own space — beats every text-space
selector, and for a cheap decoder it also costs less than embedding the corpus.
Neither is true when the decoder is a video model, an image model behind a meter,
or anything else you pay per call. There the practitioner's question is not
"should I decode?" but **"I can afford to decode m of my N candidates; what do I
get?"**

This module answers it. For a decode budget `m`, decode a uniformly random `m` of
the pool, run the artifact-space greedy selection restricted to those, and score
the chosen `k` in the artifact's space. Reported as the fraction of the oracle's
advantage over random that a budget of `m` recovers:

    recovered(m) = (partial(m) - random) / (oracle - random)

which is 0 when decoding buys nothing and 1 when a partial decode is as good as
decoding everything. The text-space selector is scored beside it, since it is
what a pipeline does when it decodes nothing at all.

    python3 budget.py code
    python3 budget.py sql
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

BUDGETS = [10, 20, 30, 50]
FRACS = [0.05, 0.10, 0.20, 0.40, 0.70]
SEEDS = 40


def loaders(name: str):
    """(texts, artifact distance, coverage fn) for a decoder."""
    if name == "code":
        import domain_code as D
        rows = [json.loads(l) for l in
                (D.OUT / "executed.jsonl").read_text().splitlines()]
        seen, uniq = set(), []
        for r in rows:
            if r["src"].strip() not in seen:
                seen.add(r["src"].strip())
                uniq.append(r)
        fps = [r["fp"] for r in uniq]
        return [r["src"] for r in uniq], D.behaviour_cells(fps)
    if name == "sql":
        import domain_sql as D
        rows = [json.loads(l) for l in
                (D.OUT / "executed.jsonl").read_text().splitlines()]
        seen, uniq = set(), []
        for r in rows:
            k = " ".join(r["sql"].split()).lower()
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        res = [r["rows"] for r in uniq]
        return [r["sql"] for r in uniq], D.row_coverage(res)
    if name == "regex":
        import domain_regex as D
        rows = [json.loads(l) for l in
                (D.OUT / "executed.jsonl").read_text().splitlines()]
        seen, uniq = set(), []
        for r in rows:
            if r["pattern"] not in seen:
                seen.add(r["pattern"])
                uniq.append(r)
        hits = [r["hits"] for r in uniq]
        return [r["pattern"] for r in uniq], D.match_coverage(hits)
    raise SystemExit(f"unknown decoder {name}")


def partial(cov, n: int, k: int, m: int, seed: int) -> float:
    """Decode a random m of the pool, then select greedily among those."""
    rng = np.random.default_rng(seed)
    sub = list(rng.choice(n, size=min(m, n), replace=False))
    if len(sub) <= k:
        return cov(sub)
    picks = P.greedy_oracle(lambda ii: cov([sub[i] for i in ii]),
                            len(sub), k, 0, seed)
    return cov([sub[i] for i in picks])


def main() -> None:
    from pipeline import embed
    name = sys.argv[1] if len(sys.argv) > 1 else "code"
    texts, cov = loaders(name)
    n = len(texts)
    TD = P.pairwise_cosine(embed(texts))
    print(f"[{name}] {n} candidates")

    out = {"decoder": name, "n": n, "rows": []}
    print("\n  fraction of the oracle's advantage over random that a decode")
    print("  budget of m recovers (mean of {} seeds)\n".format(SEEDS))
    hdr = "   k   random  maxmin   oracle" + "".join(
        f"{'m=' + str(int(100*f)) + '%':>9}" for f in FRACS)
    print(hdr)
    for k in BUDGETS:
        if k >= n:
            continue
        rnd = float(np.mean([cov(P.random_pick(n, k, s)) for s in range(SEEDS)]))
        mm = float(np.mean([cov(P.greedy_maxmin(TD, k, s)) for s in range(SEEDS)]))
        orc = float(np.mean([cov(P.greedy_oracle(cov, n, k, 150, s))
                             for s in range(3)]))
        span = orc - rnd
        cells = []
        row = {"k": k, "random": rnd, "maxmin": mm, "oracle": orc, "partial": {}}
        for f in FRACS:
            m = max(k, int(round(f * n)))
            v = float(np.mean([partial(cov, n, k, m, s) for s in range(SEEDS)]))
            row["partial"][str(f)] = {"m": m, "coverage": v,
                                      "recovered": (v - rnd) / span if span else None}
            cells.append(f"{100*(v-rnd)/span:>8.0f}%" if span else "      --")
        out["rows"].append(row)
        print(f"  {k:>3} {rnd:>8.0f}{mm:>8.0f}{orc:>9.0f}" + "".join(cells))
        print(f"      {'':>8}{100*(mm-rnd)/span:>7.0f}%{'':>9}" if span else "",
              end="\n" if span else "")

    print("\n  same rows as raw coverage")
    print(hdr)
    for row in out["rows"]:
        cells = [f"{row['partial'][str(f)]['coverage']:>9.0f}" for f in FRACS]
        print(f"  {row['k']:>3} {row['random']:>8.0f}{row['maxmin']:>8.0f}"
              f"{row['oracle']:>9.0f}" + "".join(cells))

    (HERE / "runs" / f"budget_{name}.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote runs/budget_{name}.json")

    # the practitioner's summary: smallest m reaching 70% of the ceiling
    print("\n  smallest decode budget reaching 70% of the oracle's advantage")
    for row in out["rows"]:
        hit = next((f for f in FRACS
                    if (row["partial"][str(f)]["recovered"] or 0) >= 0.70), None)
        m = row["partial"][str(hit)]["m"] if hit else None
        print(f"    k={row['k']:<4} " +
              (f"m={m} ({100*hit:.0f}% of the pool, {m/row['k']:.1f}x the "
               f"selection budget)" if hit else "not reached at 70% of the pool"))


if __name__ == "__main__":
    main()
