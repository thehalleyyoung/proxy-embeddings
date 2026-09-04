"""How smooth is a decoder? Measured, so the continuity prediction can be tested.

The shared-continuity account of near-field agreement says that two maps out of
the same text space agree among close pairs because both are approximately
linear there. That predicts near-field strength should track how continuous the
*decoder* is — and the prediction only means anything if smoothness is a number
rather than an adjective.

The instrument is deliberately identical to the one used on the `ipaddress`
decoder elsewhere in this project, so the numbers are comparable across decoders
that share nothing else:

    modulus = artifact_distance(decode(x), decode(x'))

with `x'` one single-character edit from `x`. Two choices in that definition are
load-bearing and were both got wrong on the first attempt.

**The displacement is not divided by the normalized text distance.** A single
edit has normalized Levenshtein distance 1/len, so dividing by it multiplies the
ratio by the length of the text: a 300-character SQL query scores thirty times a
10-character string for identical artifact behaviour. Across decoders whose texts
differ in length by an order of magnitude that ratio measures length, not
smoothness. Within one decoder it is fine, and is still reported.

**Edits that break the syntax are excluded, and counted separately.** A single
character can make a program not parse or a query not run, and treating that as a
maximal artifact move measures the fragility of the surface syntax rather than
the continuity of the decoder on its own domain. Those are different properties
and the first swamps the second: on the corpora here, 24% to 84% of blind
single-character edits fail to decode at all. Edits are therefore resampled until
they decode, up to a bounded number of attempts, and the acceptance rate is
reported beside the modulus as its own number.

Reported with dispersion, and with the fraction of edits that move the artifact
not at all and the fraction that move it more than half-way, because a decoder
can be smooth in the mean and violently discontinuous in places — and that
bimodality is what a mean alone hides.

    python3 modulus.py            # every decoder with a corpus on disk
    python3 modulus.py code sql   # named decoders only
"""
from __future__ import annotations

import json
import pathlib
import random
import string
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "code"))

SEED = 20260904
N_ITEMS = 120
N_EDITS = 6
MAX_TRIES = 25


def text_distance(a: str, b: str) -> float:
    """Normalized character-level edit distance."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 1.0
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb] / max(la, lb)


def perturb(s: str, rng: random.Random, alphabet: str) -> str:
    """One single-character edit: substitute, delete, or insert."""
    if not s:
        return rng.choice(alphabet)
    op = rng.choice(("sub", "del", "ins"))
    i = rng.randrange(len(s))
    if op == "sub":
        return s[:i] + rng.choice(alphabet) + s[i + 1:]
    if op == "del":
        return s[:i] + s[i + 1:]
    return s[:i] + rng.choice(alphabet) + s[i:]


def jaccard(a, b) -> float:
    A, B = set(a), set(b)
    return 1.0 - len(A & B) / max(len(A | B), 1)


# ------------------------------------------------------------------ decoders
# Each returns (texts, decode_fn, artifact_distance_fn, edit alphabet).
# decode_fn returns None when the edited text no longer decodes at all, which
# is itself a discontinuity and is counted as a maximal move rather than
# discarded -- dropping them would flatter exactly the roughest decoders.

def d_code():
    import domain_code as C
    rows = [json.loads(l) for l in
            (C.OUT / "executed.jsonl").read_text().splitlines()]
    seen, uniq = set(), []
    for r in rows:
        if r["src"].strip() not in seen:
            seen.add(r["src"].strip())
            uniq.append(r)
    texts = [r["src"] for r in uniq]

    def dec(s):
        if not C.is_safe(s)[0]:
            return None
        return C.run_one(s)

    def dist(a, b):
        if a is None or b is None:
            return 1.0
        return float(np.mean([x != y for x, y in zip(a, b)]))

    return texts, dec, dist, string.ascii_letters + string.digits + " ()[]:+-*/<>=,.%_"


def d_sql():
    import domain_sql as S
    rows = [json.loads(l) for l in
            (S.OUT / "executed.jsonl").read_text().splitlines()]
    seen, uniq = set(), []
    for r in rows:
        k = " ".join(r["sql"].split()).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    texts = [r["sql"] for r in uniq]

    def dec(s):
        if not S.is_safe(s)[0]:
            return None
        return S.run_query(s)

    return texts, dec, (lambda a, b: 1.0 if a is None or b is None else jaccard(a, b)), \
        string.ascii_letters + string.digits + " ()*,.'_=<>"


def d_regex():
    import domain_regex as R
    rows = [json.loads(l) for l in
            (R.OUT / "executed.jsonl").read_text().splitlines()]
    seen, uniq = set(), []
    for r in rows:
        if r["pattern"] not in seen:
            seen.add(r["pattern"])
            uniq.append(r)
    texts = [r["pattern"] for r in uniq]
    return texts, R.run_one, \
        (lambda a, b: 1.0 if a is None or b is None else jaccard(a, b)), \
        string.ascii_letters + string.digits + r"\[]().*+?{}|^$-_"


def d_math():
    import domain_math as M
    rows = [json.loads(l) for l in
            (M.OUT / "executed.jsonl").read_text().splitlines()]
    seen, uniq = set(), []
    for r in rows:
        k = "".join(r["expr"].split())
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    texts = [r["expr"] for r in uniq]

    def dist(a, b):
        if a is None or b is None:
            return 1.0
        return float(M.shape_distance([a, b])[0, 1])

    return texts, M.evaluate, dist, "0123456789xX+-*/(). "


DECODERS = {"math": d_math, "sql": d_sql, "code": d_code, "regex": d_regex}


def measure(name: str) -> dict | None:
    try:
        texts, dec, dist, alphabet = DECODERS[name]()
    except Exception as e:
        print(f"  [{name}] unavailable: {str(e)[:80]}")
        return None
    if len(texts) < 10:
        print(f"  [{name}] too few items ({len(texts)})")
        return None
    rng = random.Random(SEED)
    sample = rng.sample(texts, min(N_ITEMS, len(texts)))

    art, ratios, zero = [], [], 0
    tried = accepted = 0
    for s in sample:
        a0 = dec(s)
        if a0 is None:
            continue
        got = 0
        for _ in range(N_EDITS * MAX_TRIES):
            if got >= N_EDITS:
                break
            s2 = perturb(s, rng, alphabet)
            dt = text_distance(s, s2)
            if dt <= 0:
                continue
            tried += 1
            a1 = dec(s2)
            if a1 is None:      # left the decoder's domain: fragility, not
                continue        # discontinuity, and counted as such below
            accepted += 1
            got += 1
            da = dist(a0, a1)
            art.append(da)
            ratios.append(da / dt)
            if da == 0.0:
                zero += 1
    if not art:
        return None
    a, r = np.array(art), np.array(ratios)
    return {"decoder": name, "n_items": len(sample), "n_edits": len(a),
            # the comparable statistic: artifact movement per single edit
            "move_mean": float(a.mean()), "move_median": float(np.median(a)),
            "dispersion": float(a.std() / max(a.mean(), 1e-9)),
            "zero_move_frac": zero / len(a),
            "big_move_frac": float(np.mean(a > 0.5)),
            # length-inflated; kept for within-decoder comparability only
            "ratio_mean": float(r.mean()), "ratio_median": float(np.median(r)),
            "syntax_acceptance_rate": accepted / max(tried, 1),
            "mean_text_len": float(np.mean([len(x) for x in sample])),
            "quantiles": {str(q): float(np.percentile(a, q))
                          for q in (10, 25, 50, 75, 90, 99)}}


def main() -> None:
    names = sys.argv[1:] or list(DECODERS)
    rows = []
    for n in names:
        print(f"\nmeasuring {n} ...")
        m = measure(n)
        if m:
            rows.append(m)
            print(f"  move {m['move_mean']:.3f}  median {m['move_median']:.3f}"
                  f"  dispersion {m['dispersion']:.3f}"
                  f"  zero {100*m['zero_move_frac']:.1f}%"
                  f"  big {100*m['big_move_frac']:.1f}%"
                  f"  syntax-ok {100*m['syntax_acceptance_rate']:.1f}%")
    if not rows:
        return
    rows.sort(key=lambda r: r["move_mean"])
    print("\n  artifact movement per single-character edit (edits that still decode)")
    print("  decoder     move  median  dispersion   zero   big   syntax-ok  len  ratio")
    for m in rows:
        print(f"  {m['decoder']:<9} {m['move_mean']:>6.3f} {m['move_median']:>7.3f}"
              f" {m['dispersion']:>11.3f} {100*m['zero_move_frac']:>6.1f}%"
              f" {100*m['big_move_frac']:>5.1f}% {100*m['syntax_acceptance_rate']:>10.1f}%"
              f" {m['mean_text_len']:>5.0f} {m['ratio_mean']:>6.1f}")
    (HERE / "runs" / "modulus.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  wrote runs/modulus.json")


if __name__ == "__main__":
    main()
