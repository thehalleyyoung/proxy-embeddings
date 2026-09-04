"""
The axis-realization audit, ported to psychometric items.

Images got pixel statistics and poems got prosody: a channel outside the
embedding the method optimizes, so that "the axis moved the artifact" is not
graded by the same instrument the objective is written in. For test items that
channel is the item's *psychometric* structure, and one property dominates all
the others.

DIFFICULTY IS THE POINT OF AN ITEM BANK. Test information is maximized where
item difficulty matches examinee ability, so a bank whose items cluster at one
difficulty is informative only in a narrow band of the ability continuum, no
matter how varied its content. Difficulty is a latent parameter, not a semantic
feature, so a sentence embedding is close to blind to it: two items can be far
apart in embedding space and identical in difficulty, or adjacent in embedding
space and separated by a standard deviation of ability. Nothing in the axis
scoring can see this, because nothing in the axis scoring reads anything but
the embedding.

The features below are the load-bearing determinants of difficulty for
deductive-reasoning items, drawn from what actually moves item difficulty in
practice rather than from what is easy to count:

  load         distinct named entities and stated constraints -- how much has
               to be held in working memory at once
  depth        conditional connectives and chained implications -- how many
               inference steps separate the stem from the key
  negation     negations and exception phrasing, which reliably add difficulty
               independently of content
  interference explicitly irrelevant material the solver must discard
  format       option count, stem length, presence of any non-standard
               response format

Format is included because a bank that is uniform in response format is a bank
that measures one thing in one way, whatever its content diversity says.

Usage:
    python3 item_realization.py ../real/psychometric_ihd [more corpora...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))

N_PERM = 2000
RNG = np.random.default_rng(20260901)

COND = r"\b(if|then|whenever|unless|only if|provided that|implies|therefore|hence|so that)\b"
NEG = r"\b(not|never|no|none|neither|nor|except|without|cannot|excluding|other than)\b"
QUANT = r"\b(all|some|every|each|any|exactly|at least|at most|most|few|none)\b"
IRREL = r"\b(irrelevant|unrelated|does not affect|not relevant|regardless|immaterial|unverified)\b"
MODAL = r"\b(must|could|might|may|should|necessarily|possibly)\b"
# requirement markers: each one introduces a condition the solver has to hold
REQ = (r"\b(must|required to|at least|at most|no more than|no fewer than|"
       r"without|while (?:preserving|maintaining|keeping)|avoid(?:ing)?|"
       r"prevent(?:ing)?|preserv(?:e|ing)|exactly|only if|only using|"
       r"within \d|not exceed|cannot exceed|subject to)\b")


def split_stem_options(text: str) -> tuple[str, list[str]]:
    m = list(re.finditer(r"^\s*([A-Z])[.)]\s+(.*)$", text, re.M))
    if not m:
        return text, []
    return text[:m[0].start()], [x.group(2) for x in m]


def item_features(text: str) -> dict[str, float]:
    stem, opts = split_stem_options(text)
    words = re.findall(r"[A-Za-z']+", stem)
    nw = max(len(words), 1)
    # capitalized tokens that are not sentence-initial: a usable proxy for the
    # named entities a solver has to track
    ents = set(re.findall(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", stem, re.M))
    # Constraints: requirement-bearing clauses, not punctuation. Counting
    # semicolons and conditionals alone undercounts badly -- a stem that says
    # "must keep availability above 99.9%, prevent data loss, avoid schema
    # changes, preserve forensic logs, and use only reversible changes" states
    # five constraints with no semicolon and no "if" in sight, and scored 0.
    constraints = (len(re.findall(r";", stem))
                   + len(re.findall(COND, stem, re.I))
                   + len(re.findall(REQ, stem, re.I)))
    opt_lens = [len(o) for o in opts] or [0]
    return {
        "stem_chars": float(len(stem)),
        "stem_words": float(nw),
        "n_options": float(len(opts)),
        "option_len_mean": float(np.mean(opt_lens)),
        "option_len_sd": float(np.std(opt_lens)),
        "n_entities": float(len(ents)),
        "n_constraints": float(constraints),
        "conditional_rate": len(re.findall(COND, stem, re.I)) / nw * 100,
        "negation_rate": len(re.findall(NEG, stem, re.I)) / nw * 100,
        "quantifier_rate": len(re.findall(QUANT, stem, re.I)) / nw * 100,
        "modal_rate": len(re.findall(MODAL, stem, re.I)) / nw * 100,
        "irrelevance_markers": float(len(re.findall(IRREL, stem, re.I))),
        "has_numbers": float(bool(re.search(r"\d", stem))),
        "has_symbols": float(bool(re.search(r"[→∧∨¬↑←→↓×]", text))),
    }


# ---- difficulty index: the composite the bank is graded on -----------------
LOAD = ["n_entities", "n_constraints"]
DEPTH = ["conditional_rate", "stem_words"]
# `irrelevance_markers` was measured at exactly 0.00 in every item of both a
# difficulty-conditioned and an unconditioned bank, including items commanded at
# "Integration under interference". The generator plants irrelevant detail
# without labelling it -- three availability zones, a Tuesday, fourteen
# engineers -- and the regex only catches explicit markers ("irrelevant",
# "does not affect"). Detecting unlabelled irrelevance needs a semantic judge,
# not a pattern, so the feature is excluded from the composite rather than left
# in contributing a constant.
CONF = ["negation_rate"]


def difficulty_index(F: np.ndarray, names: list[str]) -> np.ndarray:
    """Standardized sum of the load, depth and confusion blocks.

    Not calibrated against examinees -- it is an ordering, not a b-parameter.
    Its purpose is to show whether the corpus SPANS difficulty, which is a
    question about spread and does not need an absolute scale.
    """
    Z = (F - F.mean(0)) / np.clip(F.std(0), 1e-9, None)
    idx = {n: i for i, n in enumerate(names)}
    blocks = [np.mean([Z[:, idx[c]] for c in blk], axis=0)
              for blk in (LOAD, DEPTH, CONF) if all(c in idx for c in blk)]
    return np.mean(blocks, axis=0)


def eta_squared(X: np.ndarray, labels: list[str]) -> float:
    X = X.reshape(len(X), -1)
    lab = np.array(labels)
    grand = X.mean(axis=0, keepdims=True)
    ss_tot = float(((X - grand) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    ss_b = 0.0
    for u in set(labels):
        g = X[lab == u]
        if len(g):
            ss_b += len(g) * float(((g.mean(axis=0, keepdims=True) - grand) ** 2).sum())
    return ss_b / ss_tot


def perm_p(X, labels, observed, n: int = N_PERM) -> float:
    if not np.isfinite(observed):
        return float("nan")
    lab = np.array(labels)
    hits = sum(eta_squared(X, list(RNG.permutation(lab))) >= observed for _ in range(n))
    return float((hits + 1) / (n + 1))


def audit(corpus_dir: Path, limit: int = 600) -> dict:
    recs = [json.loads(l) for l in
            (corpus_dir / "corpus.jsonl").read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("text")][:limit]
    feats = [item_features(r["text"]) for r in recs]
    names = sorted(feats[0])
    F = np.array([[f[k] for k in names] for f in feats])
    Fz = (F - F.mean(0)) / np.clip(F.std(0), 1e-9, None)
    diff = difficulty_index(F, names)

    out = {"corpus": corpus_dir.name, "n": len(recs),
           "difficulty_sd": float(diff.std()),
           "difficulty_range": [float(diff.min()), float(diff.max())],
           "features": {}, "axes": []}
    for j, n_ in enumerate(names):
        v = F[:, j]
        out["features"][n_] = {"mean": round(float(v.mean()), 3),
                               "cv": round(float(v.std() / max(abs(v.mean()), 1e-9)), 3)}

    if recs and recs[0].get("spec"):
        by: dict[str, list[str]] = {}
        for r in recs:
            for k, v in r["spec"].items():
                by.setdefault(k, []).append(str(v))
        for axis in sorted(by):
            idx = [i for i, r in enumerate(recs) if axis in r["spec"]]
            labels = [str(recs[i]["spec"][axis]) for i in idx]
            if len(idx) < 20 or len(set(labels)) < 2:
                continue
            e_all = eta_squared(Fz[idx], labels)
            e_dif = eta_squared(diff[idx, None], labels)
            out["axes"].append({
                "axis": axis, "levels": len(set(labels)), "n": len(idx),
                "structure_eta2": round(float(e_all), 4),
                "structure_p": round(perm_p(Fz[idx], labels, e_all), 4),
                "difficulty_eta2": round(float(e_dif), 4),
                "difficulty_p": round(perm_p(diff[idx, None], labels, e_dif), 4),
            })
        out["axes"].sort(key=lambda a: -a["difficulty_eta2"])
    return out


def report(res: dict) -> str:
    L = [f"\n=== {res['corpus']} (n={res['n']}) ===",
         f"difficulty index: sd {res['difficulty_sd']:.3f}, "
         f"range [{res['difficulty_range'][0]:.2f}, {res['difficulty_range'][1]:.2f}]",
         f"\n{'axis':<36}{'lvl':>4}{'struct η²':>11}{'p':>7}{'diff η²':>9}{'p':>7}"]
    for a in res["axes"]:
        star = "*" if a["difficulty_p"] < 0.05 else " "
        L.append(f"{star}{a['axis'][:35]:<35}{a['levels']:>4}{a['structure_eta2']:>11.4f}"
                 f"{a['structure_p']:>7.3f}{a['difficulty_eta2']:>9.4f}{a['difficulty_p']:>7.3f}")
    flat = sorted(res["features"].items(), key=lambda kv: abs(kv[1]["cv"]))[:6]
    L.append("\nfeatures the corpus barely varies (lowest cv):")
    for k, v in flat:
        L.append(f"   {k:<22} mean {v['mean']:>8.2f}   cv {v['cv']:>6.3f}")
    L.append("  * = axis moves the difficulty index (p<0.05)")
    return "\n".join(L)


if __name__ == "__main__":
    dirs = [Path(d) for d in sys.argv[1:]] or [RESEARCH / "real" / "psychometric_ihd"]
    out = []
    for d in dirs:
        r = audit(d.resolve())
        out.append(r)
        print(report(r), flush=True)
    json.dump(out, open(HERE / "item_realization.json", "w"), indent=2)
    print(f"\nwrote {HERE / 'item_realization.json'}")
