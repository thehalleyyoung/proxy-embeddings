"""
Key-position balance in a generated item bank, and whether a ceiling reading is real.

Two things fell out of the difficulty measurement and both need separating from
their confounds.

First, every item in both banks was answered identically on all five samples,
which reads as a ceiling: the solver is far above the bank's difficulty and the
consistency signal has no resolution left. That is only credible if the solver
is actually tracking item content rather than defaulting to a position.

Second, the answers skew heavily toward A. That is either a property of the
BANK -- generated items whose key sits in position A far more often than chance,
which is a real psychometric defect, since a test-wise examinee scores above
chance by always answering A -- or a property of the SOLVER, which would make
the difficulty measurement meaningless rather than merely saturated.

The permutation test separates them. Ask each item twice: once as written, once
with its options randomly reordered. Record which OPTION TEXT is chosen each
time, not which letter.

  content-tracking   the same option text is chosen both times. The solver reads
                     the item; an A-skew in the original ordering is then a fact
                     about where the generator puts its keys.
  position-locked    the same LETTER is chosen both times while the text under it
                     changes. The solver is not reading; every difficulty number
                     derived from it is void.

Usage:
    python3 item_keybias.py ../real/psychometric_ihd ../real/psychometric_naive
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

from item_difficulty import ask  # noqa: E402
from item_realization import split_stem_options  # noqa: E402

N_ITEMS = 40
WORKERS = 12
LETTERS = "ABCDEFGH"


def render(stem: str, options: list[str]) -> str:
    body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{stem.rstrip()}\n{body}"


def probe(rec: dict, rng: random.Random) -> dict | None:
    stem, opts = split_stem_options(rec["text"])
    if len(opts) < 3:
        return None
    order = list(range(len(opts)))
    perm = order[:]
    rng.shuffle(perm)
    if perm == order:
        perm = perm[1:] + perm[:1]
    a1 = ask(render(stem, opts))
    a2 = ask(render(stem, [opts[i] for i in perm]))
    if not a1 or not a2:
        return None
    i1, i2 = LETTERS.find(a1), LETTERS.find(a2)
    if not (0 <= i1 < len(opts) and 0 <= i2 < len(perm)):
        return None
    return {
        "orig_letter": a1,
        "orig_index": i1,
        # which ORIGINAL option the second answer landed on
        "perm_maps_to_orig_index": perm[i2],
        "same_content": perm[i2] == i1,
        "same_letter": i1 == i2,
        "n_options": len(opts),
    }


def run(corpus_dir: Path, n_items: int = N_ITEMS) -> dict:
    recs = [json.loads(l) for l in
            (corpus_dir / "corpus.jsonl").read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("text") and re.search(r"^\s*A[.)]", r["text"], re.M)]
    rng = random.Random(20260901)
    idx = rng.sample(range(len(recs)), min(n_items, len(recs)))
    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(probe, recs[i], random.Random(1000 + i)) for i in idx]
        for f in as_completed(futs):
            r = f.result()
            if r:
                rows.append(r)
    if not rows:
        return {"corpus": corpus_dir.name, "n": 0}
    pos = Counter(r["orig_letter"] for r in rows)
    n = len(rows)
    k = int(np.median([r["n_options"] for r in rows]))
    exp = n / k
    chi2 = sum((pos.get(LETTERS[i], 0) - exp) ** 2 / exp for i in range(k))
    return {
        "corpus": corpus_dir.name, "n": n, "n_options": k,
        "position_counts": {LETTERS[i]: pos.get(LETTERS[i], 0) for i in range(k)},
        "position_share": {LETTERS[i]: round(pos.get(LETTERS[i], 0) / n, 3) for i in range(k)},
        "chi2_vs_uniform": round(float(chi2), 2),
        "df": k - 1,
        "content_tracking_rate": round(float(np.mean([r["same_content"] for r in rows])), 3),
        "letter_locked_rate": round(float(np.mean([r["same_letter"] for r in rows])), 3),
    }


if __name__ == "__main__":
    dirs = [Path(d) for d in sys.argv[1:]] or [RESEARCH / "real" / "psychometric_ihd"]
    out = []
    for d in dirs:
        r = run(d.resolve())
        out.append(r)
        print(f"\n=== {r['corpus']} (n={r['n']}) ===")
        if not r["n"]:
            continue
        print(f"  key position, as answered: {r['position_share']}")
        print(f"  chi2 vs uniform = {r['chi2_vs_uniform']} on {r['df']} df "
              f"(critical 7.81 at p=.05 for 3 df)")
        print(f"  answer follows CONTENT under permutation: {r['content_tracking_rate']:.1%}")
        print(f"  answer stays on same LETTER:              {r['letter_locked_rate']:.1%}",
              flush=True)
    json.dump(out, open(HERE / "item_keybias.json", "w"), indent=2)
    print(f"\nwrote {HERE / 'item_keybias.json'}")
