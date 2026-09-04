"""Implementation of the public API. Import from `decodergap` instead."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import _core as _P


# ------------------------------------------------------------------ 1. audit

@dataclass
class Audit:
    profile: object
    control: dict
    dedup: dict
    purity: dict
    selectors: dict | None
    verdicts: dict = field(default_factory=dict)

    def summary(self) -> str:
        v = self.verdicts
        out = ["decodergap audit", "-" * 60]
        c = self.control
        out.append(f"decoder near-minus-far      {c['decoder']:+.3f}")
        out.append(f"encoder-only baselines      "
                   f"{', '.join(f'{b:+.3f}' for b in c['baselines'])}")
        out.append(f"clears every baseline       {'YES' if c['clears'] else 'NO'}")
        if not c["clears"]:
            out.append("  -> a correlation between your text distances and your")
            out.append("     artifact distances is not evidence here: two encoders")
            out.append("     that never saw the decoder agree at least as well.")
        for k in ("deduplication_aggregate", "deduplication_per_item",
                  "max_min_selection"):
            if k in v:
                out.append(f"{k:<28}{v[k].get('verdict')}")
        return "\n".join(out)


def audit(texts, embed, decode=None, distance=None, coverage=None,
          artifact_distance=None, n_bins: int = 10, budgets=(10, 20, 30, 50),
          extra_encoders=None) -> Audit:
    """Is text-space distance machinery sound for this decoder?

    `embed(texts) -> array`. Either supply `decode` and `distance`, or a
    precomputed `artifact_distance` matrix. `coverage(indices) -> float`, scored
    in the artifact's space, enables the selector comparison.

    The decoder-free control is not optional and is not a refinement: the
    near-field agreement that appears to qualify a proxy is reproduced by two
    encoders with no decoder involved, so the decoder profile is only evidence
    insofar as it clears the whole distribution of encoder-to-encoder baselines.
    """
    E = np.asarray(embed(texts))
    TD = _P.pairwise_cosine(E)
    if artifact_distance is None:
        arts = [decode(t) for t in texts]
        n = len(arts)
        AD = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                AD[i, j] = AD[j, i] = distance(arts[i], arts[j])
    else:
        AD = np.asarray(artifact_distance)

    prof = _P.near_field_profile(None, AD, text_dist=TD, n_bins=n_bins)
    views = {"primary": TD}
    for k, enc in (extra_encoders or _default_encoders()).items():
        try:
            views[k] = _P.pairwise_cosine(np.asarray(enc(texts)))
        except Exception:
            pass

    def nmf(A, B):
        p = _P.near_field_profile(None, B, text_dist=A, n_bins=n_bins)
        return p.near_minus_far

    names = [k for k in views if k != "primary"]
    baselines = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            baselines.append(nmf(views[names[i]], views[names[j]]))
        baselines.append(nmf(views["primary"], views[names[i]]))
    baselines = [b for b in baselines if np.isfinite(b)]
    control = {"decoder": float(prof.near_minus_far),
               "baselines": [float(b) for b in baselines],
               "clears": bool(baselines and prof.near_minus_far > max(baselines))}

    dd = _P.dedup_quality(TD, AD)
    pur = _P.reject_purity(TD, AD)
    sel = (_P.selector_comparison(TD, coverage, [b for b in budgets if b < len(texts)])
           if coverage else None)
    v = _P.decision_verdicts(prof, dd, sel, pur)
    if not control["clears"]:
        v["correlational_validation"] = {
            "verdict": "CONFOUNDED",
            "why": ("the decoder profile does not exceed the encoder-to-encoder "
                    "baselines, so agreement between text distance and artifact "
                    "distance is explained without reference to the decoder")}
    return Audit(prof, control, dd, pur, sel, v)


def _default_encoders():
    def char(texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            min_df=2, max_features=20000).fit_transform(texts).toarray()
        return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

    def word(texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z_]+",
                            ngram_range=(1, 2), min_df=2).fit_transform(texts).toarray()
        return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return {"char_tfidf": char, "word_tfidf": word}


# ----------------------------------------------------------------- 2. triage

_EXACT = re.compile(r"\bexactly\b|\bprecisely\b|\bequals?\b|==|"
                    r"\b\d+(\.\d+)?\s*(seconds?|frames?|px|pixels?|items?|rows?)\b", re.I)
_BAND = re.compile(r"\bbetween\b|\bat least\b|\bat most\b|\bwithin\b|±|\+/-|"
                   r"\bapproximately\b|\baround\b|\broughly\b", re.I)
# Quantities that come OUT of a generation rather than being written into it:
# durations, counts, aggregates, and holistic qualities. A duration is the
# clearest case -- nothing in a prompt writes 2.4 seconds into a shot, the shot
# turns out that long.
_EMERGENT = re.compile(r"\b(pacing|rhythm|mood|tone|style|feel|balance|overall|"
                       r"average|total|count|distribution|ratio|complexity|"
                       r"difficulty|quality|coherence|aggregate|sum|"
                       r"duration|lasts?|tempo|speed|bpm|number of|how many|"
                       r"brightness|contrast|saturation)\b"
                       r"|\b\d+(\.\d+)?\s*(seconds?|minutes?|frames?|px|pixels?)\b"
                       r"|\b(length|size|area) of\b", re.I)
_CONSTRUCT = re.compile(r"\b(contains?|includes?|mentions?|must have|shows?|"
                        r"depicts?|uses?|named|labelled|labeled|colou?r|"
                        r"appears?|present)\b", re.I)


@dataclass
class Triage:
    target: str
    kind: str
    exactness: str
    notes: list
    suggestion: str | None

    def __str__(self) -> str:
        s = [f"target      {self.target[:70]}",
             f"kind        {self.kind}",
             f"stated as   {self.exactness}"]
        s += [f"note        {n}" for n in self.notes]
        if self.suggestion:
            s.append(f"suggest     {self.suggestion}")
        return "\n".join(s)


def triage(target: str) -> Triage:
    """Classify a target on the three axes that govern whether stating it works.

    Heuristic and text-based: it reads the wording of the requirement, so it is
    a prompt for the practitioner's own judgement rather than a measurement.
    The axes it is checking were each isolated by a separate experiment.
    """
    notes, sug = [], None
    emergent = bool(_EMERGENT.search(target))
    constructible = bool(_CONSTRUCT.search(target)) and not emergent
    kind = ("constructible" if constructible else
            "emergent" if emergent else "unclear")
    exact = "band" if _BAND.search(target) else (
        "exact" if _EXACT.search(target) else "unclear")

    if kind == "constructible":
        notes.append("satisfiable by writing something specific into the artifact; "
                     "compliance was high and flat in rarity where measured")
    elif kind == "emergent":
        notes.append("must come out of the generation rather than being written "
                     "in; compliance decays into the generator's tail")
    else:
        notes.append("could not classify from the wording: ask whether satisfying "
                     "this needs a specific token written, or a behaviour found")

    if exact == "exact" and kind != "constructible":
        sug = ("restate as a range. On an emergent decoder this moved compliance "
               "from 27.3% to 78.8% in the rare band. Put the tolerance in the "
               "PROMPT: relaxing acceptance afterwards recovered nothing, because "
               "failures were wholesale rather than near-misses.")
    elif exact == "band":
        notes.append("stated as a band, which is the cheap intervention already "
                     "applied; note the price -- a band buys range compliance at "
                     "the cost of point compliance")

    notes.append("check observability separately: if your acceptance criterion is "
                 "computed somewhere the generator cannot see -- including a "
                 "quantization you apply after generation -- it cannot verify its "
                 "own compliance, and that alone cost us a registered prediction")
    return Triage(target, kind, exact, notes, sug)


def completion_check(arms: dict, tol: float = 0.15) -> dict:
    """Refuse a cross-arm verdict when the arms did not complete at equal rates.

    `arms` maps an arm name to a list of per-generation records, each truthy if
    that generation returned a usable artifact. Attrition that correlates with
    the treatment manufactures effects: in this project it produced a 19-point
    tolerance "benefit" that vanished once completion was equalized, and in a
    collaborating experiment a boundary result resting on five observations out
    of sixty. In both cases the tell was the sample count, not the effect size.

    This check is cheaper than the reasoning that recovers from either.
    """
    rates = {k: (sum(1 for x in v if x) / len(v) if v else 0.0)
             for k, v in arms.items()}
    spread = max(rates.values()) - min(rates.values()) if rates else 0.0
    ok = spread <= tol
    return {"rates": rates, "spread": float(spread), "ok": bool(ok),
            "verdict": ("comparable" if ok else
                        f"REFUSED: completion rates differ by "
                        f"{100*spread:.1f} points across arms, so a difference in "
                        f"outcome cannot be separated from a difference in how "
                        f"often each arm answered at all. Report intent-to-treat "
                        f"and conditional figures separately, or equalize the "
                        f"generation budget until the rates match.")}


# -------------------------------------------------------------- 2b. misses

def diagnose_misses(misses, axis_distance) -> dict:
    """Which axis is binding? Read it off the shape of the miss distribution.

    `misses` is any iterable of failed generations; `axis_distance(m) -> dict`
    returns, for one miss, how far it fell on each axis of the acceptance
    criterion, plus the key `present` saying whether the required thing appeared
    at all.

    The two failure profiles measured in this work point to opposite repairs and
    a single compliance number cannot distinguish them:

      misses just outside the criterion  -> the criterion is at a resolution the
                                            generator cannot verify; widen it or
                                            state the tolerance
      misses spread out, or absent       -> the target is emergent; tolerance
                                            will not save it, decompose it or
                                            budget for the attempt rate
    """
    rows = [axis_distance(m) for m in misses]
    if not rows:
        return {"n": 0}
    axes = [k for k in rows[0] if k != "present"]
    present = float(np.mean([bool(r.get("present", True)) for r in rows]))
    near = [r for r in rows if r.get("present", True)]
    out = {"n": len(rows), "required_thing_present": present, "axes": {}}
    for a in axes:
        v = [r[a] for r in near if r.get(a) is not None]
        if v:
            out["axes"][a] = {"median": float(np.median(v)),
                              "frac_exact": float(np.mean(np.array(v) == 0)),
                              "frac_within_1": float(np.mean(np.array(v) <= 1))}
    if near and axes:
        out["frac_within_1_on_all_axes"] = float(np.mean(
            [all((r.get(a) or 0) <= 1 for a in axes) for r in near]))
    w = out.get("frac_within_1_on_all_axes", 0.0)
    if present < 0.5:
        out["verdict"] = "EMERGENT: the required thing is usually not produced at all"
    elif w >= 0.5:
        out["verdict"] = ("UNOBSERVABLE CRITERION: most misses are within one step "
                          "of the target on every axis, so the artifact is already "
                          "nearly right and the acceptance test is finer than the "
                          "generator can verify")
    else:
        out["verdict"] = ("EMERGENT: misses are spread rather than adjacent, so "
                          "widening the criterion will not recover them")
    return out


# ------------------------------------------------------------------- 3. plan

def plan(decode_budget: int, pool: int, select: int) -> str:
    """What a decode budget buys, from the measured curves.

    The curve is stable as a share of the POOL and unstable as a multiple of the
    selection budget (4.7x to 12.1x for the same 70% across our decoders), so
    the fraction is what is reported. These are two decoders' worth of curve and
    should be treated as a shape rather than as constants: running `audit` on
    your own decoder replaces them with yours.
    """
    frac = decode_budget / max(pool, 1)
    ratio = decode_budget / max(select, 1)
    # measured recovered-fraction, mean across the two decoders, by pool share
    grid = {0.05: 0.07, 0.10: 0.23, 0.20: 0.43, 0.40: 0.68, 0.70: 0.88}
    xs = sorted(grid)
    rec = float(np.interp(frac, xs, [grid[x] for x in xs]))
    lines = [f"decode budget {decode_budget} of {pool} candidates "
             f"({100*frac:.0f}% of the pool), selecting {select}",
             f"  expected recovery of the full-decode advantage: ~{100*rec:.0f}%",
             f"  decode budget is {ratio:.1f}x the selection budget"]
    if ratio < 3:
        lines.append("  WARNING: with fewer than ~3 decoded candidates per pick, "
                     "greedy selection has almost nothing to choose between and "
                     "recovers close to nothing. Select fewer items, or decode more.")
    if rec < 0.36:
        lines.append("  at this budget text-space selection is competitive, which "
                     "is a statement about how little either buys, not about the "
                     "text space being sound.")
    else:
        lines.append("  this exceeds what text-space selection bought on any "
                     "decoder we measured (its ceiling was 36% of the advantage).")
    lines.append("  the recovered fraction falls monotonically in the number "
                 "selected at every decode budget.")
    return "\n".join(lines)
