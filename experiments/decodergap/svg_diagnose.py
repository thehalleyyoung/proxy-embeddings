"""Why did the SVG targets miss? Tag presence first, then quantization distance.

The registered SVG prediction failed at 18.6% against a floor of 70%. One
candidate explanation is that an SVG target is a conjunction of four quantized
properties, three of which are the scoring apparatus rather than anything the
generator can evaluate -- so the misses would be quantization misses, near the
target on every axis but not exactly on it.

A tolerance rescore cannot test that, because a one-band rescore lifts compliance
under the rival story too if enough mass happens to sit nearby. The distributions
can, and one of them decides it faster than the rest:

  **Tag presence.** Was the required element type drawn AT ALL, anywhere? If it
  is usually absent, the generator is not drawing what it was told and every
  distance computed downstream is a distance between the target and marks that
  were never candidates for it. Quantization cannot explain an absent element.

Only if the tag is usually present do the other three distributions mean
anything: Chebyshev distance on the 6x6 grid, size-band distance, hue-band
distance, each measured over marks that share the required tag.

    python3 svg_diagnose.py
"""
from __future__ import annotations

import collections, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "runs" / "lever_svg"


def parse(cell):
    tag, rest = cell.split("@", 1)
    pos, sz, hue = rest.split("|")
    gx, gy = (int(v) for v in pos.split(","))
    return tag, gx, gy, int(sz[1:]), int(hue[1:])


def main():
    rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["arm"] == "instruct" and "produced" in r]
    if not rows:
        raise SystemExit("no instruct rows with recorded marks; re-run lever.py run svg")
    print(f"{len(rows)} instruct generations with recorded marks")

    hits = [r for r in rows if r.get("hit")]
    miss = [r for r in rows if not r.get("hit")]
    print(f"  hits {len(hits)}  misses {len(miss)}\n")

    tag_present, gd, sd, hd, n_marks = 0, [], [], [], []
    for r in miss:
        ttag, tgx, tgy, tsz, thue = parse(r["cell"])
        prod = [parse(c) for c in r["produced"]]
        n_marks.append(len(prod))
        same = [p for p in prod if p[0] == ttag]
        if same:
            tag_present += 1
            gd.append(min(max(abs(p[1] - tgx), abs(p[2] - tgy)) for p in same))
            sd.append(min(abs(p[3] - tsz) for p in same))
            hd.append(min(abs(p[4] - thue) for p in same))

    n = len(miss)
    print("  THE DECIDING NUMBER")
    print(f"    misses in which the required TAG was drawn at all: "
          f"{tag_present}/{n} = {100*tag_present/max(n,1):.1f}%")
    print(f"    median marks drawn per generation: {int(np.median(n_marks)) if n_marks else 0}\n")

    if tag_present < 0.5 * n:
        print("    -> the required element is usually ABSENT. The generator is not")
        print("       drawing what it was told, so quantization does not explain")
        print("       the failure and the distances below are over marks that were")
        print("       never candidates.\n")
    else:
        print("    -> the required element is usually PRESENT, so the distances")
        print("       below are meaningful.\n")

    def dist(name, vals, cap):
        if not vals:
            print(f"    {name}: no data")
            return
        c = collections.Counter(vals)
        tot = len(vals)
        line = "  ".join(f"{k}:{100*c[k]/tot:.0f}%" for k in sorted(c) if k <= cap)
        print(f"    {name:<22} median {np.median(vals):.0f}   {line}")

    print("  MISS DISTANCE, over misses where the tag was drawn")
    dist("grid cells (0-5)", gd, 5)
    dist("size bands (0-4)", sd, 4)
    dist("hue bands (0-8)", hd, 8)

    within1 = sum(1 for a, b, c in zip(gd, sd, hd) if a <= 1 and b <= 1 and c <= 1)
    if gd:
        print(f"\n    misses within ONE band on all three axes: "
              f"{within1}/{len(gd)} = {100*within1/len(gd):.1f}% of tag-present misses"
              f" ({100*within1/max(n,1):.1f}% of all misses)")
    json.dump({"n_miss": n, "tag_present": tag_present,
               "grid": gd, "size": sd, "hue": hd, "within1": within1},
              open(OUT / "diagnosis.json", "w"))
    print(f"\n  wrote {OUT/'diagnosis.json'}")


if __name__ == "__main__":
    main()
