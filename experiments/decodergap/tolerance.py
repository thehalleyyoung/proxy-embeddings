"""Does stating a target as a tolerance instead of an exact value help?

The programs decoder has the hardest targets in the study: a required return
value has to come out of whatever the function computes, and compliance decays
into the tail. A collaborating measurement on a different decoder found that
loosening an exact requirement into a band moved compliance from 0% to 45.8%
with nothing else changed, which — if it holds here — is the most directly
actionable thing in the paper for anyone steering a generator they cannot see
inside.

Same targets, same decoder, same generator, one difference: the requirement is
stated as `must return exactly V` or as `must return a value between V-d and
V+d`. A tolerant hit is scored tolerantly and an exact hit exactly, so the two
arms are not scored on the same event; the comparison is between two ways of
ASKING, and the looser ask is also a looser success criterion. That is the
honest framing, because it is also the choice a practitioner faces: accept a
range, or insist on a point.

    python3 tolerance.py run 3
"""
from __future__ import annotations

import json, pathlib, random, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "runs" / "tolerance2"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "code"))

TOL = 0.10          # +/- 10% of |V|, minimum 2


def band(v: int) -> tuple[int, int]:
    d = max(2, int(round(TOL * abs(v))))
    return v - d, v + d


def prompt_exact(t) -> str:
    import domain_code as C
    return (C.PROMPT.replace(
        "Reply with the function only",
        f"**Requirement: `f({t['input']})` must return exactly "
        f"{t['want_str']}.** The function must still be a genuine "
        f"general-purpose function, not a lookup table.\n"
        "Reply with the function only"))


def prompt_tol(t) -> str:
    import domain_code as C
    lo, hi = band(int(t["want_str"]))
    return (C.PROMPT.replace(
        "Reply with the function only",
        f"**Requirement: `f({t['input']})` must return a value between "
        f"{lo} and {hi} inclusive.** The function must still be a genuine "
        f"general-purpose function, not a lookup table.\n"
        "Reply with the function only"))


def _one(args):
    import domain_code as C
    from pipeline import chat
    arm, t, seed = args
    rec = {"arm": arm, "band": t["band"], "rarity": t["rarity"], "seed": seed}
    try:
        p = prompt_exact(t) if arm == "exact" else prompt_tol(t)
        txt = chat([{"role": "user", "content": p}], temperature=1.0, max_tokens=2600)
        if not (txt or "").strip():
            txt = chat([{"role": "user", "content": p}], temperature=1.0, max_tokens=4000)
        src = C.extract(txt or "")
        if src and C.is_safe(src)[0]:
            fp = C.run_one(src)
            if fp:
                got = fp[t["input_idx"]]
                rec["got"] = got
                want = int(t["want_str"])
                if got.startswith("I") and got[1:].lstrip("-").isdigit():
                    v = int(got[1:])
                    lo, hi = band(want)
                    rec["hit_exact"] = (v == want)
                    rec["hit_tol"] = (lo <= v <= hi)
                else:
                    rec["hit_exact"] = rec["hit_tol"] = False
    except Exception as e:
        rec["error"] = str(e)[:160]
    return rec


def run(n_seeds=3):
    tg = json.loads((HERE / "runs" / "steer" / "targets.json").read_text())
    tg = [t for t in tg if t["want"].startswith("I")
          and t["want"][1:].lstrip("-").isdigit()]
    jobs = [(a, t, s) for a in ("exact", "tolerant") for t in tg
            for s in range(n_seeds)]
    print(f"{len(jobs)} jobs over {len(tg)} integer targets")
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(_one, jobs))
    with (OUT / "results.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    report(rows)


def report(rows=None):
    if rows is None:
        rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines()]
    def ci(h):
        h = np.array(h, dtype=float); rng = np.random.default_rng(0)
        bs = [rng.choice(h, len(h)).mean() for _ in range(4000)]
        return h.mean(), np.percentile(bs, [2.5, 97.5]), len(h)
    print("\n  how the target is ASKED, and how it is SCORED")
    for arm, key, label in (("exact", "hit_exact", "asked exactly, scored exactly"),
                            ("tolerant", "hit_tol", "asked as a band, scored on the band"),
                            ("tolerant", "hit_exact", "asked as a band, scored exactly"),
                            ("exact", "hit_tol", "asked exactly, scored on the band")):
        h = [bool(r.get(key)) for r in rows if r["arm"] == arm]
        if not h: continue
        m,(lo,hi),n = ci(h)
        print(f"    {label:<40} {100*m:5.1f}%  [{100*lo:4.1f},{100*hi:5.1f}]  n={n}")
    print("\n  by rarity band, asked-as-a-band / scored-on-band")
    bands = sorted({r["band"] for r in rows}, key=lambda b: float(b.split(",")[0].lstrip("[")))
    for b in bands:
        e = [bool(r.get("hit_exact")) for r in rows if r["arm"]=="exact" and r["band"]==b]
        tl = [bool(r.get("hit_tol")) for r in rows if r["arm"]=="tolerant" and r["band"]==b]
        if e and tl:
            print(f"    {b:<22} exact {100*np.mean(e):5.1f}%   tolerant {100*np.mean(tl):5.1f}%"
                  f"   lift {np.mean(tl)-np.mean(e):+.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        report()
