"""
Within-level stereotypy in the image domain, which iteration 33 left unmeasured.

The stereotypy scan needs enough items per axis level to read one, and the image
runs have sixty prompts across eleven axes -- too few, so images came back
"unmeasured" rather than clean. That is a gap in the evidence, not a finding,
and it is cheap to close: the conditioning that a stereotypy scan reads lands in
the WRITTEN INSTRUCTION, and instructions are text. No renders are needed.

So this generates instructions from the steer7 axes at a level count the scan can
actually score, and runs iteration 33's validated metric on them. The question is
whether the coloured-card pattern -- an axis obeyed through one lexical device --
appears in a domain whose conditioning must cross into a second generator, or
whether the seam that costs images 35% of their conditioning (§6.14) also
scatters the device.

Usage:
    python3 image_stereotypy.py [n]
"""
from __future__ import annotations

import collections
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

from pipeline import USAGE, chat                     # noqa: E402
from stereotypy_scan import words                    # noqa: E402
from probe_image_seam import final_axes              # noqa: E402

SEED = 20260914
N_DEFAULT = 160
WORKERS = 8
RUN = RESEARCH / "real" / "dalle_steer7_maxmin"

# The contract block only, matching the seam probe: no avoid-block, no mined
# attractors, no structural bans, so that any concentration measured is the
# axis set's own and not the corpus-state machinery's.
WRITE = """Write ONE image-generation instruction for a post-modern artwork. \
One paragraph, 30-70 words, concrete and visual.

Let these choices shape the work. Express each as something visible, never as \
abstract art-theory:
{contracts}

Name the medium, the composition, the palette, the surface, and the treatment. \
Output only the instruction."""


def gen(spec: dict) -> str | None:
    contracts = "\n".join(f"  - {k}: {v}" for k, v in spec.items())
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.2) + 2.0 * attempt)
        try:
            t = chat([{"role": "user", "content": WRITE.format(contracts=contracts)}],
                     temperature=1.0, max_tokens=600)
            if len(t.strip()) > 40:
                return t.strip()
        except Exception:
            pass
    return None


def build(axes: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    used = {a: collections.Counter() for a in axes}
    out = []
    for _ in range(n):
        s = {}
        for a, lv in axes.items():
            lo = min(used[a].get(x, 0) for x in lv)
            s[a] = rng.choice([x for x in lv if used[a].get(x, 0) == lo])
            used[a][s[a]] += 1
        out.append(s)
    return out


CEILING = 0.90   # see stereotypy_scan: a near-universal word is a constant


def score(rows: list[dict], axis: str, min_items: int = 12) -> dict:
    docs = [words(r["text"]) for r in rows]
    echo = [set().union(*[words(str(v)) for v in r["spec"].values()]) for r in rows]
    labs = [str(r["spec"][axis]) for r in rows]
    corpus = collections.Counter(w for d in docs for w in d)
    n_all = len(docs)
    out = {}
    for lv in sorted(set(labs)):
        idx = [i for i, l in enumerate(labs) if l == lv]
        if len(idx) < min_items:
            continue
        df = collections.Counter()
        for i in idx:
            df.update(docs[i] - echo[i])
        for w in [w for w in df if corpus[w] / n_all > CEILING]:
            del df[w]
        if df:
            w, c = df.most_common(1)[0]
            out[lv] = {"device": w, "share": c / len(idx), "n": len(idx)}
    return out


def main(n: int) -> None:
    axes = final_axes(RUN)
    specs = build(axes, n, SEED)
    print(f"{len(axes)} axes; generating {n} instructions "
          f"(~{n // max(len(next(iter(axes.values()))), 1)} per level)", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        outs = list(ex.map(gen, specs))
    rows = [{"spec": s, "text": t} for t, s in zip(outs, specs) if t]
    print(f"  {len(rows)}/{n} written", flush=True)

    print(f"\n{'axis':<42}{'lvl':>4}{'mean':>7}{'worst':>7}   most stereotyped level / device")
    res = []
    for a in sorted(axes):
        sc = score(rows, a)
        if len(sc) < 2:
            continue
        m = float(np.mean([x["share"] for x in sc.values()]))
        top = max(sc.items(), key=lambda kv: kv[1]["share"])
        res.append({"axis": a, "levels": len(sc), "mean": m,
                    "worst_level": top[0], "worst_device": top[1]["device"],
                    "worst_share": top[1]["share"]})
        print(f"{a[:41]:<42}{len(sc):>4}{m:>7.2f}{top[1]['share']:>7.2f}   "
              f"{top[0][:30]} / {top[1]['device']}")
    if res:
        print(f"\n{'image prompts':<20}{len(res):>4} axes   "
              f"mean device share {np.mean([r['mean'] for r in res]):.3f}   "
              f"worst {max(r['worst_share'] for r in res):.3f}")
        print("\nfor comparison (iteration 33): poems 0.533 / 0.811, "
              "psych items 0.412 / 0.727")
    json.dump(res, open(HERE / "image_stereotypy.json", "w"), indent=2)
    with open(HERE / "image_stereotypy_corpus.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT)
