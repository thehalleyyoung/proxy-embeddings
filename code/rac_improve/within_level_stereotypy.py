"""
An axis can be obeyed and still be realized the same way every time.

Every realization audit in this work asks whether a commanded level MOVES the
artifact. On the psychometric bank the answer for `Irrelevant-information
treatment` is emphatically yes: items told to carry no irrelevant information
use a colour word 7% of the time, items told to carry some use one 52-62% of
the time. The axis is obeyed.

It is obeyed through one device. Half the corpus (49.7% of 1,999 items) contains
a colour word, and reading the items shows why -- the red herring is nearly
always a coloured card, a star, a printed dot. "The cards are red and blue, and
a small star is printed on the blue card." "The symbols are printed in black on
a blue card." An axis level realized through a single lexical device is a defect
no realization audit can see, because realization is measured BETWEEN levels and
this is a collapse WITHIN one.

It matters here more than it would elsewhere. Irrelevant information is a
functioning part of an item only while the examinee cannot recognize it by its
form. A bank whose red herring is always a colour sentence teaches candidates to
skip colour sentences, and the distractor stops measuring anything -- classic
construct-irrelevant variance.

The fix follows the full-partition prescription: name the devices. An axis whose
levels enumerate the FORMS a red herring can take -- a mismatched unit, an
authority's opinion, a superseded figure, an unused constraint -- has no unnamed
remainder to fall back on, and level balancing commands each equally often.

  base      six psychometric axes, irrelevant-information treatment among them
  device    the same six plus an axis enumerating eight red-herring forms

Both arms are scored for which device carries the irrelevant detail by a judge
seeing one item at a time, blind to arm and to any axis, and BOTH are checked
for whether the original axis is still obeyed -- a fix that spreads the devices
while breaking the contract would be no fix.

Usage:
    python3 within_level_stereotypy.py [n_per_arm]
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

from pipeline import USAGE, chat, parse_json      # noqa: E402

SEED = 20260908
N_DEFAULT = 90
WORKERS = 8

BASE_AXES = {
    "Primary cognitive operation": [
        "Apply an explicit rule", "Evaluate competing explanations",
        "Model causal dependencies", "Induce a latent rule",
        "Map relational structure analogically"],
    "Required mental representation": [
        "Propositional relations", "Ordered sequence",
        "Quantitative magnitude model", "Spatial configuration",
        "Matrix of attributes", "Graph of connections"],
    "Scenario framing": [
        "Unfolding narrative", "Dialogue with conflicting claims",
        "Decontextualized symbolic prompt", "Counterfactual scenario",
        "Observed-event report"],
    "Distractor logic": [
        "Overgeneralized rule", "Single-step omissions",
        "Surface-feature matches", "Valid-looking but inconsistent",
        "Reversed relation"],
    "Constraint interaction load": [
        "Single independent constraint", "Cross-dependent constraints",
        "Several independent constraints", "Nested constraints",
        "Global all-at-once consistency"],
    "Irrelevant-information treatment": [
        "Interleaved irrelevant detail", "Salient unrelated detail",
        "Separated benign detail", "Superficially correlated detail"],
}

DEVICES = {
    "appearance": "a colour, marking, sticker or other physical appearance detail",
    "unit": "a quantity in a mismatched or irrelevant unit",
    "authority": "someone's opinion, credential or endorsement",
    "superseded": "a figure or rule that has been replaced and no longer applies",
    "unused_constraint": "an extra stated constraint that no option violates",
    "timing": "a date, duration or time of day that bears on nothing",
    "provenance": "where the information came from, or who recorded it",
    "ambience": "weather, location or sensory setting detail",
}

DEVICE_AXIS = {
    "Irrelevant-information device": [
        f"{k}: {v}" for k, v in DEVICES.items()]
}

WRITE = """Write one multiple-choice reasoning item for a cognitive assessment.

The item MUST exhibit these design behaviors:
{contracts}

Requirements: one stem, exactly 4 options (A-D), exactly one defensibly correct \
answer. The irrelevant information must be genuinely irrelevant: removing it \
must not change the answer. Return JSON only:
{{"stem": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, \
"correct": "A"}}"""

JUDGE = """You are analysing one multiple-choice reasoning item.

The item contains some detail that is IRRELEVANT -- removing it would not change \
the correct answer. What FORM does that irrelevant detail take? Answer with \
exactly one label:
{labels}
none (the item contains no irrelevant detail)

Item:
{item}

Return JSON only: {{"device": "..."}}"""

CONTRACT_JUDGE = """You are checking one multiple-choice reasoning item.

Does the item contain a detail that is genuinely irrelevant -- that is, removing \
it would NOT change which option is correct?

Item:
{item}

Return JSON only: {{"has_irrelevant": true or false}}"""


def _ask(prompt: str, key: str, ok):
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.0) + 1.5 * attempt)
        try:
            raw = chat([{"role": "user", "content": prompt}], temperature=0.0,
                       max_tokens=2000, json_mode=True)
            v = parse_json(raw)[key]
            if ok(v):
                return v
        except Exception:
            pass
    return None


def gen(spec: dict) -> str | None:
    contracts = "\n".join(f"  - {k}: {v}" for k, v in spec.items())
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.2) + 2.0 * attempt)
        try:
            o = parse_json(chat([{"role": "user",
                                  "content": WRITE.format(contracts=contracts)}],
                                temperature=1.0, max_tokens=3000, json_mode=True))
            t = o["stem"] + "\n" + "\n".join(
                f"{k}. {v}" for k, v in sorted(o["options"].items()))
            if len(t) > 80:
                return t
        except Exception:
            pass
    return None


def device_of(text: str) -> str | None:
    labels = "\n".join(f"{k} ({v})" for k, v in DEVICES.items())
    return _ask(JUDGE.format(labels=labels, item=text[:2500]), "device",
                lambda v: str(v).strip().lower() in set(DEVICES) | {"none"})


def has_irrelevant(text: str) -> bool | None:
    return _ask(CONTRACT_JUDGE.format(item=text[:2500]), "has_irrelevant",
                lambda v: isinstance(v, bool))


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


def ent(c: collections.Counter) -> float:
    keys = list(DEVICES)
    p = np.array([c.get(k, 0) for k in keys], dtype=float)
    p = p / max(p.sum(), 1)
    q = p[p > 0]
    return float(-(q * np.log(q)).sum() / np.log(len(keys)))


def main(n: int) -> None:
    arms = {"base": BASE_AXES, "device": {**BASE_AXES, **DEVICE_AXIS}}
    store, res = {}, {}
    for name, axes in arms.items():
        print(f"\narm '{name}': {n} items, {len(axes)} axes", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            texts = [t for t in ex.map(gen, build(axes, n, SEED)) if t]
        print(f"  {len(texts)}/{n} generated; judging device + contract blind",
              flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            devs = list(ex.map(device_of, texts))
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            keeps = list(ex.map(has_irrelevant, texts))
        store[name] = (texts, devs, keeps)
        d = collections.Counter(x for x in devs if x and x != "none")
        res[name] = d
        tot = max(sum(d.values()), 1)
        ok = [k for k in keeps if k is not None]
        print(f"  contract honoured (irrelevant detail present): "
              f"{sum(ok)}/{len(ok)} = {sum(ok)/max(len(ok),1):.0%}")
        print(f"  devices used: {len(d)}/8   entropy {ent(d):.3f}   "
              f"dominant {d.most_common(1)[0][0]} {d.most_common(1)[0][1]/tot:.0%}")
        print(f"    {', '.join(f'{k}:{v}' for k, v in d.most_common())}")

    a = [x for x in store["base"][1] if x and x != "none"]
    b = [x for x in store["device"][1] if x and x != "none"]
    rng = np.random.default_rng(SEED)
    de, dm = [], []
    for _ in range(8000):
        x = collections.Counter(rng.choice(a, len(a)))
        y = collections.Counter(rng.choice(b, len(b)))
        de.append(ent(y) - ent(x))
        dm.append(max(y.values()) / sum(y.values()) - max(x.values()) / sum(x.values()))
    print()
    for lab, v in (("device entropy", np.array(de)), ("dominant share", np.array(dm))):
        lo, hi = np.percentile(v, [2.5, 97.5])
        s = "*" if lo > 0 or hi < 0 else " "
        print(f"{s} device - base, {lab:<16}{v.mean():>+8.3f}  [{lo:+.3f}, {hi:+.3f}]")
    json.dump({k: dict(v) for k, v in res.items()},
              open(HERE / "within_level_stereotypy.json", "w"), indent=2)
    with open(HERE / "stereotypy_arms.jsonl", "w") as fh:
        for nm, (ts, ds, ks) in store.items():
            for t, d, k in zip(ts, ds, ks):
                fh.write(json.dumps({"arm": nm, "device": d,
                                     "has_irrelevant": k, "text": t}) + "\n")
    print(f"\nspend ${USAGE.cost_usd():.2f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT)
