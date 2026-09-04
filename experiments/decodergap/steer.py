"""Steering an output modality you cannot address, through a channel you can.

The general problem behind every result in this project: a generator is
addressable only in text, the thing you want exists in another modality, and a
decoder you do not control sits between them. Coverage and diversity are one
instance. The question underneath is *when does the text channel let you hit a
target in the artifact's space at all*, and how that depends on where the target
sits in the generator's own artifact distribution.

This module measures it as a task rather than as a correlation, because on this
project every correlation-shaped claim has been overturned by a control and
every direct measurement of a decision has held.

**Setup.** The natural artifact distribution is estimated from a corpus the
generator produced under a fixed prompt. A *target* is one behavioural cell — a
required output for a specified input — and its **rarity** is the fraction of
that natural corpus which already satisfies it. Rarity 0.4 is a target four
programs in ten hit by accident; rarity 0.0 is one nothing in the corpus hits.

**Arms.** For each target, at matched generator budget:
  `blind`      the fixed prompt, unchanged; the target is never mentioned.
  `decoy`      a DIFFERENT target stated with identical form and specificity,
               scored against the real one. This is the control that matters:
               without it, `instruct` beating `blind` is confounded by the fact
               that adding any concrete requirement to a prompt changes what the
               generator produces. Only the gap between `instruct` and `decoy`
               is evidence that the channel carried this target rather than
               merely carrying something.
  `instruct`   the target rendered into the text channel as a requirement.
  `retry`      instruct, and on failure re-prompt with the wrong answer shown.

**What is measured.** Compliance: did the decoded artifact actually satisfy the
target, checked by execution rather than by asking the model. Reported against
target rarity, which turns the result into a curve rather than a single rate:
how far into its own tail can a generator be pushed through a proxy channel.

    python3 steer.py targets      # build the target set from the natural corpus
    python3 steer.py run 240      # run all arms
    python3 steer.py report
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "runs" / "steer2"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "code"))

SEED = 20260904
N_RARITY_BANDS = 5
TARGETS_PER_BAND = 12


def _fmt(v: str) -> str:
    """The fingerprint encoding back into something a prompt can state."""
    if v.startswith("I"):
        return v[1:]
    if v.startswith("B"):
        return v[1:]
    if v.startswith("F"):
        return v[1:]
    if v.startswith("E:"):
        return f"raise {v[2:]}"
    return v


def build_targets() -> None:
    """Targets stratified by how often the natural corpus already hits them."""
    import domain_code as C
    rows = [json.loads(l) for l in
            (C.OUT / "executed.jsonl").read_text().splitlines()]
    seen, uniq = set(), []
    for r in rows:
        if r["src"].strip() not in seen:
            seen.add(r["src"].strip())
            uniq.append(r)
    fps = [r["fp"] for r in uniq]
    n = len(fps)
    print(f"natural corpus: {n} distinct programs")

    # every (input index, output value) cell, with the share of the corpus that
    # produces it -- this is the generator's own artifact distribution
    counts: dict[tuple[int, str], int] = collections.Counter()
    for fp in fps:
        for j, v in enumerate(fp):
            counts[(j, v)] += 1

    cells = [(j, v, c / n) for (j, v), c in counts.items()]
    # exclude error outputs and the empty-list input: a target should be a
    # thing a program can be asked for, not a crash or a degenerate case
    cells = [(j, v, r) for j, v, r in cells
             if not v.startswith("E:") and len(C.BATTERY[j]) >= 2]
    print(f"{len(cells)} candidate cells")

    rng = random.Random(SEED)
    bands = [(0.001, 0.02), (0.02, 0.10), (0.10, 0.30), (0.30, 1.01)]
    targets = []

    # Every cell above was observed at least once, so the rarest band the corpus
    # can supply is 1/n. The interesting end of the curve is past that: outputs
    # the generator has NEVER produced for an input. These are constructed to be
    # reachable in principle -- an integer inside the range of values the corpus
    # does produce for that input -- so a failure is the channel's, not the
    # task's.
    by_input: dict[int, set[str]] = collections.defaultdict(set)
    for j, v, _ in cells:
        by_input[j].add(v)
    idxs = [j for j in by_input if len(C.BATTERY[j]) >= 3]
    rng.shuffle(idxs)
    n_unobs = 0
    for j in idxs:
        seen_v = {int(v[1:]) for v in by_input[j]
                  if v.startswith("I") and v[1:].lstrip("-").isdigit()}
        if len(seen_v) < 4:
            continue
        lo, hi = min(seen_v), max(seen_v)
        cand = [x for x in range(lo, hi + 1) if x not in seen_v]
        if not cand:
            continue
        want = rng.choice(cand)
        targets.append({"input_idx": j, "input": C.BATTERY[j],
                        "want": f"I{want}", "want_str": str(want),
                        "rarity": 0.0, "band": "[0,0.001) unobserved"})
        n_unobs += 1
        if n_unobs >= TARGETS_PER_BAND:
            break
    print(f"  band [0,0.001) unobserved: {n_unobs} constructed")
    for lo, hi in bands:
        pool = [c for c in cells if lo <= c[2] < hi]
        rng.shuffle(pool)
        for j, v, r in pool[:TARGETS_PER_BAND]:
            targets.append({"input_idx": j, "input": C.BATTERY[j],
                            "want": v, "want_str": _fmt(v), "rarity": r,
                            "band": f"[{lo:g},{hi:g})"})
        print(f"  band [{lo:g},{hi:g}): {len(pool)} available, "
              f"{min(len(pool), TARGETS_PER_BAND)} taken")
    (OUT / "targets.json").write_text(json.dumps(targets, indent=2))
    print(f"wrote {len(targets)} targets")


BASE = (
    "Write one Python function with exactly this signature:\n\n"
    "    def f(xs):\n\n"
    "`xs` is a list of integers. Return a single integer. The function must be "
    "pure: no imports, no printing, no randomness, no I/O, no global state. "
    "Make it do something specific and interesting with the list. "
    "Reply with the function only, in a ```python code block, no commentary."
)


def instruct_prompt(t: dict) -> str:
    return (
        "Write one Python function with exactly this signature:\n\n"
        "    def f(xs):\n\n"
        "`xs` is a list of integers. Return a single integer. The function must "
        "be pure: no imports, no printing, no randomness, no I/O, no global "
        "state. Make it do something specific and interesting with the list.\n\n"
        f"**Requirement: `f({t['input']})` must return exactly "
        f"{t['want_str']}.**\n\n"
        "The function must still be a genuine general-purpose function, not a "
        "lookup table or a special case for that one input.\n"
        "Reply with the function only, in a ```python code block, no commentary."
    )


def retry_prompt(t: dict, got: str) -> str:
    return (
        instruct_prompt(t)
        + f"\n\nA previous attempt returned {_fmt(got)} for that input instead "
          f"of {t['want_str']}. Do not repeat it."
    )


def _one(args) -> dict:
    import domain_code as C
    from pipeline import chat
    arm, t, seed, decoy = args
    rec = {"arm": arm, "band": t["band"], "rarity": t["rarity"],
           "input_idx": t["input_idx"], "want": t["want"], "seed": seed}
    try:
        if arm == "blind":
            prompt = BASE
        elif arm == "decoy":
            prompt = instruct_prompt(decoy)   # same form, wrong target
        else:
            prompt = instruct_prompt(t)
        txt = chat([{"role": "user", "content": prompt}],
                   temperature=1.0, max_tokens=2600)
        if not (txt or "").strip():
            txt = chat([{"role": "user", "content": prompt}],
                       temperature=1.0, max_tokens=4000)
        src = C.extract(txt or "")
        if src and C.is_safe(src)[0]:
            fp = C.run_one(src)
            if fp:
                rec["got"] = fp[t["input_idx"]]
                rec["hit"] = (fp[t["input_idx"]] == t["want"])
        if arm == "retry" and not rec.get("hit") and rec.get("got"):
            txt = chat([{"role": "user", "content": retry_prompt(t, rec["got"])}],
                       temperature=1.0, max_tokens=2600)
            src = C.extract(txt or "")
            if src and C.is_safe(src)[0]:
                fp = C.run_one(src)
                if fp:
                    rec["got2"] = fp[t["input_idx"]]
                    rec["hit"] = (fp[t["input_idx"]] == t["want"])
            rec["calls"] = 2
        rec.setdefault("calls", 1)
    except Exception as e:
        rec["error"] = str(e)[:160]
    return rec


def run(n_seeds: int = 3) -> None:
    targets = json.loads((OUT / "targets.json").read_text())
    rng = random.Random(SEED)
    # each target's decoy is another target on the SAME input, so the two
    # prompts differ only in the required value; where no other value is
    # available for that input, a target on a different input is used.
    def pick_decoy(t):
        same = [u for u in targets
                if u["input_idx"] == t["input_idx"] and u["want"] != t["want"]]
        pool = same or [u for u in targets if u["want"] != t["want"]]
        return rng.choice(pool)
    jobs = [(arm, t, s, pick_decoy(t))
            for arm in ("blind", "decoy", "instruct", "retry")
            for t in targets
            for s in range(n_seeds)]
    print(f"{len(jobs)} generator jobs "
          f"({len(targets)} targets x 4 arms x {n_seeds} seeds)")
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(_one, jobs))
    with (OUT / "results.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} results")


def report() -> None:
    rows = [json.loads(l) for l in
            (OUT / "results.jsonl").read_text().splitlines()]
    bands = sorted({r["band"] for r in rows},
                   key=lambda b: float(b.split(",")[0].lstrip("[")))
    arms = ["blind", "decoy", "instruct", "retry"]

    def rate(sel):
        h = [bool(r.get("hit")) for r in sel]
        if not h:
            return float("nan"), 0, (float("nan"), float("nan"))
        p = float(np.mean(h))
        rng = np.random.default_rng(0)
        bs = [np.mean(rng.choice(h, len(h))) for _ in range(2000)]
        return p, len(h), tuple(np.percentile(bs, [2.5, 97.5]))

    print("\n  compliance: did the decoded artifact actually hit the target?")
    print("  rarity band        n   " + "".join(f"{a:>22}" for a in arms))
    for b in bands:
        line = f"  {b:<16}"
        n0 = None
        cells = []
        for a in arms:
            p, n, ci = rate([r for r in rows if r["band"] == b and r["arm"] == a])
            n0 = n0 or n
            cells.append(f"{100*p:>10.1f}% [{100*ci[0]:.0f},{100*ci[1]:.0f}]")
        print(line + f"{n0:>4}   " + "".join(f"{c:>22}" for c in cells))

    print("\n  overall")
    for a in arms:
        p, n, ci = rate([r for r in rows if r["arm"] == a])
        calls = sum(r.get("calls", 1) for r in rows if r["arm"] == a)
        print(f"    {a:<10} {100*p:>6.1f}%  [{100*ci[0]:.1f}, {100*ci[1]:.1f}]"
              f"   n={n}  calls={calls}  hits/call={p*n/max(calls,1):.3f}")

    sel = [r for r in rows if r["arm"] in ("decoy", "instruct")]
    xs = np.array([r["rarity"] for r in sel])
    ys = np.array([bool(r.get("hit")) for r in sel])
    ai = np.array([r["arm"] == "instruct" for r in sel])
    print("\n  lift over the DECOY control (same prompt form, wrong target)")
    for lo, hi in ((-0.1, 0.001), (0.001, 0.02), (0.02, 0.10), (0.10, 0.30), (0.30, 1.01)):
        m = (xs >= lo) & (xs < hi)
        if m.sum() < 6:
            continue
        b = ys[m & ~ai].mean() if (m & ~ai).sum() else np.nan
        i = ys[m & ai].mean() if (m & ai).sum() else np.nan
        print(f"    rarity [{lo:g},{hi:g})   decoy {100*b:5.1f}%  "
              f"instruct {100*i:5.1f}%   lift {i-b:+.3f}")
    json.dump({"n_rows": len(rows)}, open(OUT / "summary.json", "w"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "targets":
        build_targets()
    elif cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        report()
