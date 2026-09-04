"""
The identifiability question, asked of corpora that actually differ.

The first attempt compared six poem arms that shared an axis set and differed by
one intervention each. Kendall tau between representations came out at +0.067,
which looks like a decisive answer -- until the control: subsampling within each
representation showed between-arm spread SMALLER than within-arm sampling noise
in three of four views (ratios 0.76, 0.64, 0.69, with prosody marginal at 1.07).
Arms that a representation cannot tell apart will rank randomly under it, so a
near-zero tau was the expected result whatever the truth about identifiability.
The experiment could not have answered the question it asked.

This asks it of corpora built by genuinely different generation methods, where
the paper's own competitive results establish that large differences exist:
the psychometric bank from this method against naive sampling, persona
prompting, self-instruct, evol-instruct and high-temperature sampling.

The separability control now runs FIRST and gates the reading. A representation
whose between-corpus spread does not exceed its within-corpus noise is reported
as unable to rank, and excluded from the tau -- rather than contributing a
random ordering to an average that would then look like disagreement.

Usage:
    python3 identifiability_v2.py [n_per_corpus]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

import is_diversity_identified as M   # noqa: E402

SEED = 20260918
N_DEFAULT = 200
N_DRAWS = 120
SUB = 0.75

CORPORA = {
    "rac": "psychometric_ihd",
    "naive": "psychometric_naive",
    "persona": "psychometric_persona",
    "self-inst": "psychometric_self_instruct",
    "evol-inst": "psychometric_evol_instruct",
    "high-temp": "psychometric_high_temp",
}


def load(n: int) -> dict[str, list[str]]:
    rng = random.Random(SEED)
    out = {}
    for name, d in CORPORA.items():
        p = RESEARCH / "real" / d / "corpus.jsonl"
        if not p.exists():
            continue
        texts = [json.loads(l).get("text", "") for l in
                 p.read_text().splitlines() if l.strip()]
        texts = [t for t in texts if len(t) > 120]
        if len(texts) >= n:
            out[name] = rng.sample(texts, n)
    return out


def main(n: int) -> None:
    corp = load(n)
    print(f"{len(corp)} corpora, {n} items each: {', '.join(corp)}\n", flush=True)
    rng = np.random.default_rng(SEED)

    reps = {k: v for k, v in M.REPS.items() if k != "nomic"}
    scores, sep = {}, {}
    for rname, fn in reps.items():
        print(f"embedding under {rname} ...", flush=True)
        per = {}
        for a, ts in corp.items():
            X = fn(ts)
            vals = []
            m = int(len(X) * SUB)
            for _ in range(N_DRAWS):
                idx = rng.choice(len(X), m, replace=False)
                vals.append(M.norm_p5_nn(X[idx]))
            per[a] = (float(np.mean(vals)), float(np.std(vals)))
        means = np.array([per[a][0] for a in corp])
        sds = np.array([per[a][1] for a in corp])
        ratio = float(means.std() / max(sds.mean(), 1e-12))
        scores[rname] = {a: per[a][0] for a in corp}
        sep[rname] = ratio

    print(f"\nSEPARABILITY GATE (between-corpus spread / within-corpus noise)")
    print(f"{'representation':<14}{'between':>10}{'within':>10}{'ratio':>8}   can it rank?")
    usable = []
    for rname in reps:
        means = np.array([scores[rname][a] for a in corp])
        b = float(means.std())
        w = b / max(sep[rname], 1e-12)
        ok = sep[rname] > 1.0
        usable += [rname] if ok else []
        print(f"{rname:<14}{b:>10.4f}{w:>10.4f}{sep[rname]:>8.2f}   "
              f"{'yes' if ok else 'NO -- excluded'}")

    print(f"\nnormalized 5th-percentile NN")
    print(f"{'corpus':<12}" + "".join(f"{r:>11}" for r in reps) + f"{'worst':>10}")
    for a in corp:
        row = [scores[r][a] for r in reps]
        print(f"{a:<12}" + "".join(f"{v:>11.4f}" for v in row) + f"{min(row):>10.4f}")

    if len(usable) < 2:
        print(f"\nonly {len(usable)} representation can rank these corpora; "
              f"the identifiability question stays open.")
    else:
        from scipy.stats import kendalltau
        names = list(corp)
        taus = []
        print(f"\nKendall tau, usable representations only ({', '.join(usable)})")
        for i, r1 in enumerate(usable):
            for r2 in usable[i + 1:]:
                t, _ = kendalltau([scores[r1][a] for a in names],
                                  [scores[r2][a] for a in names])
                taus.append(t)
                print(f"  {r1:<10} vs {r2:<10}  tau = {t:+.2f}")
        print(f"\nmean tau over usable pairs: {np.mean(taus):+.3f}")
        for r in usable:
            b = max(names, key=lambda a: scores[r][a])
            w = min(names, key=lambda a: scores[r][a])
            print(f"  {r:<10} best {b:<12} worst {w}")
        bw = max(names, key=lambda a: min(scores[r][a] for r in usable))
        print(f"\nbest WORST-CASE corpus: {bw}")
    json.dump({"scores": scores, "separability": sep},
              open(HERE / "identifiability_v2.json", "w"), indent=2)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT)
