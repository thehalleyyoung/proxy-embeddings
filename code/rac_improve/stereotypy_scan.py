"""
Within-level stereotypy, measured without a hand-built device taxonomy.

Iteration 32 found the psychometric bank realizing one axis level through one
lexical device -- the red herring was nearly always a coloured card -- and
showed the defect is invisible to every realization audit here, all of which
compare BETWEEN levels. Establishing that on one axis took a judge and a
purpose-built list of eight devices. That does not scale to every axis of every
corpus, and the question worth answering is whether the defect is peculiar to
psychometric items or general to language-valued conditioning.

So this measures the same thing lexically and offline. For each axis level,
find the words most over-represented at that level relative to the corpus, and
ask what fraction of the items at that level use the single most over-
represented one. A level realized through many devices spreads its distinctive
vocabulary across many words and scores low; a level realized through one
device concentrates it and scores high.

    device(l)     = the most frequent non-echo content word among items at level l
    stereotypy(l) = P(device(l) | level l)

Three things this deliberately does NOT do, each because the first version did
and failed its own validation.

It does not correct against a permutation of level labels. That null preserves
the corpus vocabulary, so when a construct is realized monotonously ACROSS the
whole corpus -- the psychometric red herring is a colour in 49.7% of all items
-- a shuffled level looks exactly as monotonous as the real one and the null
absorbs the entire effect. The known positive scored +0.028 and ranked fourth.
Corpus-wide monotony is the defect, so a null that conditions on the corpus
cannot see it.

It does not count words that appear in the level description itself. A level
reading "Distant long shot with tiny figures" makes "shot" the top word of its
own level, which is prompt echo rather than a device, and scored 83%.

It does not score an axis with fewer than two scorable levels; several did, and
a one-level reading is not a reading.

Validation is built in and binding. The psychometric `Irrelevant-information
treatment` axis is the known positive from the judge-based measurement, where
one device carried 76% of items. If the metric does not flag it, the metric is
wrong and nothing else it reports should be believed.

Usage:
    python3 stereotypy_scan.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent

N_PERM = 60
FLOOR = 0.20          # a device must appear in >= 20% of the level's items
CEILING = 0.90        # ...and must NOT be near-universal in the corpus
MIN_ITEMS = 12        # levels smaller than this are not scored
RNG = np.random.default_rng(20260909)

STOP = set("""the a an and or of to in for with that this is are be as on at by from which must
not all any each per than then when if it its their have has can could should would may might
one two three four five who what where how why into over under about after before between both
each few more most other some such only own same so too very will just now also there here they
them we you your our but do does did done being been was were had having no nor own s t don
using use used within without across while during against toward towards upon
""".split())


def words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z'-]{2,}", t.lower()) if w not in STOP}


def echo_words(level: str) -> set[str]:
    """Words the level description itself supplies, which the generator will
    repeat back. Counting them measures prompt echo, not a realization device."""
    return words(level)


def stereotypy(docs: list[set[str]], labels: list[str]) -> dict:
    """Per-level max document frequency among the level's most-lifted words.

    Takes pre-tokenized documents: the permutation null calls this hundreds of
    times per axis, and re-tokenizing inside made the scan quadratic in the
    corpus for no reason.
    """
    n = len(docs)
    corpus_df = collections.Counter(w for d in docs for w in d)
    out = {}
    for lv in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lv]
        if len(idx) < MIN_ITEMS:
            continue
        m = len(idx)
        lvl_df = collections.Counter(w for i in idx for w in docs[i])
        echo = echo_words(lv)
        best, best_share = None, 0.0
        for w, c in lvl_df.items():
            if w in echo:
                continue
            # A word in nearly every item of the corpus is a constant, not a
            # device: it cannot distinguish one level's realization from
            # another's. Image instructions all open "Create a...", which made
            # `create` the top word of all 11 axes at 0.98-1.00 and the reading
            # degenerate. The ceiling is on CORPUS frequency, not level
            # frequency, so a genuinely corpus-wide device such as the
            # psychometric "blue" -- 48% of its level against 49.7% of the
            # corpus -- is still counted.
            if corpus_df[w] / n > CEILING:
                continue
            p = c / m
            if p > best_share:
                best, best_share = w, p
        out[lv] = {"n": m, "device": best, "share": float(best_share),
                   "lift": float(best_share / max(corpus_df[best] / n, 1e-9))
                   if best else 0.0}
    return out


def scan(name: str, recs: list[dict], text_key: str = "text") -> list[dict]:
    texts = [r[text_key] for r in recs if r.get(text_key)]
    specs = [r.get("spec") or {} for r in recs if r.get(text_key)]
    docs = [words(t) for t in texts]
    axes = sorted({k for s in specs for k in s if not str(k).startswith("_")})
    rows = []
    for a in axes:
        idx = [i for i, s in enumerate(specs) if a in s]
        if len(idx) < 2 * MIN_ITEMS:
            continue
        t = [docs[i] for i in idx]
        lab = [str(specs[i][a]) for i in idx]
        obs = stereotypy(t, lab)
        if len(obs) < 2:          # a one-level reading is not a reading
            continue
        o = float(np.mean([x["share"] for x in obs.values()]))
        top = max(obs.items(), key=lambda kv: kv[1]["share"])
        rows.append({"corpus": name, "axis": a, "levels": len(obs),
                     "n": len(idx), "mean_share": o,
                     "worst_level": top[0][:38],
                     "worst_device": top[1]["device"],
                     "worst_share": top[1]["share"],
                     "worst_lift": top[1]["lift"]})
    return rows


def load(p: Path, key: str = "text") -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    corpora: list[tuple[str, list[dict]]] = []

    psy = RESEARCH / "real" / "psychometric_ihd" / "corpus.jsonl"
    if psy.exists():
        corpora.append(("psych items", load(psy)[:900]))

    poems: list[dict] = []
    for d in sorted((RESEARCH / "live").glob("poems_ctl_exp*")):
        f = d / "corpus.jsonl"
        if f.exists():
            poems += load(f)
    if poems:
        corpora.append(("poems", poems))

    items: list[dict] = []
    for d in ("items_diff", "items", "items_nodiff"):
        f = RESEARCH / "live" / d / "corpus.jsonl"
        if f.exists():
            items += load(f)
    if items:
        corpora.append(("cloud items", items))

    # Images: the axis conditioning lands in the written instruction, so the
    # instruction is where a lexical device would show.
    log = RESEARCH / "real" / "dalle_steer7_maxmin" / "log.jsonl"
    if log.exists():
        recs = load(log)
        for r in recs:
            if isinstance(r.get("spec"), str):
                try:
                    r["spec"] = json.loads(r["spec"].replace("'", '"'))
                except Exception:
                    r["spec"] = {}
            r["text"] = r.get("prompt", "")
        corpora.append(("image prompts", recs))

    all_rows = []
    for name, recs in corpora:
        rows = scan(name, recs)
        all_rows += rows
        if not rows:
            print(f"\n=== {name}: no axis had enough items to score")
            continue
        rows.sort(key=lambda r: -r["worst_share"])
        print(f"\n=== {name} (n={len(recs)}) ===")
        print(f"{'axis':<36}{'lvl':>4}{'mean':>7}{'worst':>7}   most stereotyped level / device")
        for r in rows:
            print(f"{r['axis'][:35]:<36}{r['levels']:>4}{r['mean_share']:>7.2f}"
                  f"{r['worst_share']:>7.2f}   {r['worst_level'][:24]} / "
                  f"{r['worst_device']} (x{r['worst_lift']:.1f})")

    if all_rows:
        print(f"\n{'corpus':<16}{'axes':>6}{'mean device share':>19}{'worst':>8}")
        for name in dict.fromkeys(r["corpus"] for r in all_rows):
            g = [r for r in all_rows if r["corpus"] == name]
            print(f"{name:<16}{len(g):>6}"
                  f"{np.mean([r['mean_share'] for r in g]):>19.3f}"
                  f"{max(r['worst_share'] for r in g):>8.3f}")
        v = [r for r in all_rows if r["corpus"] == "psych items"
             and "Irrelevant" in r["axis"]]
        if v:
            r = v[0]
            ok = r["worst_share"] >= 0.40
            print(f"\nVALIDATION -- known positive "
                  f"(psych items / {r['axis']}):")
            print(f"  most stereotyped level uses '{r['worst_device']}' in "
                  f"{r['worst_share']:.0%} of its items  ->  "
                  f"{'PASS' if ok else 'FAIL'}")
            if not ok:
                print("  metric does not reproduce the judge-based finding; "
                      "do not believe the rest of this table")
    json.dump(all_rows, open(HERE / "stereotypy_scan.json", "w"), indent=2)
    print(f"\nwrote {HERE / 'stereotypy_scan.json'}")


if __name__ == "__main__":
    main()
