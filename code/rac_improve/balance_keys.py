"""
Uniform key position, without knowing the key.

The measured defect: in `real/psychometric_ihd`, 90% of items place the correct
answer in option A (chi-square 90.4 on 3 df against uniform, critical value
7.81). A test-wise examinee who always answers A scores 90% on that bank
without reading a single stem. The naive bank is imbalanced too, less severely
and in a different direction (B, 55%).

No diversity metric in this work can see this. Key position is not a semantic
property, so embedding diversity, n-gram diversity, Vendi and coverage are all
exactly as high on a bank whose key is always A as on a balanced one. The
axis set cannot see it either: every axis conditions on item *content*, and
even `Distractor logic` governs what the distractors are like rather than where
the key sits among them.

The cause is structural rather than incidental. A generator asked for a stem
and four options writes the answer it has in mind first and then constructs
distractors around it, so position A is where the key lands by default. Nothing
downstream disturbs that, because nothing downstream looks at position.

The fix does not require an answer key. Permuting an item's options uniformly
at random moves the key with its own text, whatever position it occupied, so
the resulting bank has uniform key positions by construction. This is the
standard practice in test assembly and it costs one shuffle per item.

Two details matter for correctness:

  * options whose text refers to position ("both A and B", "none of the above")
    cannot be permuted safely, so items containing them are left untouched and
    counted separately;
  * the permutation is seeded per item id, so the operation is deterministic
    and a corpus can be rebalanced reproducibly.

Usage:
    python3 balance_keys.py ../real/psychometric_ihd            # report only
    python3 balance_keys.py ../real/psychometric_ihd --write    # write corpus_balanced.jsonl
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

from item_realization import split_stem_options  # noqa: E402

LETTERS = "ABCDEFGH"
# option text that names a position cannot be reordered without breaking meaning
POSITIONAL = re.compile(
    r"\b(none of the above|all of the above|both\s+[A-H]\s+and\b|either\s+[A-H]\b"
    r"|options?\s+[A-H]\b|answers?\s+[A-H]\b|[A-H]\s+and\s+[A-H]\b)", re.I)


def permutable(options: list[str]) -> bool:
    return len(options) >= 3 and not any(POSITIONAL.search(o) for o in options)


def balance_item(text: str, seed: int) -> tuple[str, bool]:
    """Return (rewritten item, whether it was permuted)."""
    stem, opts = split_stem_options(text)
    if not permutable(opts):
        return text, False
    rng = random.Random(seed)
    order = list(range(len(opts)))
    perm = order[:]
    rng.shuffle(perm)
    if perm == order:                      # guarantee an actual move
        perm = perm[1:] + perm[:1]
    body = "\n".join(f"{LETTERS[i]}. {opts[j]}" for i, j in enumerate(perm))
    return f"{stem.rstrip()}\n{body}", True


def balance_corpus(corpus_dir: Path, write: bool = False) -> dict:
    src = corpus_dir / "corpus.jsonl"
    recs = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    n_perm = n_skip = 0
    out = []
    for r in recs:
        t = r.get("text", "")
        if not t:
            out.append(r)
            continue
        new, moved = balance_item(t, seed=int(r.get("i", 0)) + 20260901)
        n_perm += moved
        n_skip += (not moved)
        r = dict(r)
        r["text"] = new
        r["key_balanced"] = moved
        out.append(r)
    res = {"corpus": corpus_dir.name, "n": len(recs),
           "permuted": n_perm, "left_alone_positional": n_skip}
    if write:
        dst = corpus_dir / "corpus_balanced.jsonl"
        dst.write_text("".join(json.dumps(r) + "\n" for r in out))
        res["written"] = str(dst)
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    dirs = [Path(a) for a in args] or [RESEARCH / "real" / "psychometric_ihd"]
    for d in dirs:
        r = balance_corpus(d.resolve(), write=write)
        print(f"{r['corpus']}: {r['permuted']}/{r['n']} permuted, "
              f"{r['left_alone_positional']} left alone (positional option text)"
              + (f" -> {r['written']}" if write else ""))
