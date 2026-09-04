"""
Is diversity a property of the corpus, or of the embedder?

Everything in this method rests on one move: define diversity as distance in a
chosen embedding, then maximize the minimum of it. Every result this sequence
produced says the same thing about that move from a different angle --
concentration relocates into whatever the axis set does not name, an
intervention reduces a measured concentration only when it enumerates the whole
space THE MEASURE is defined over, and a corpus can be simultaneously the least
repetitive on whole-vocabulary overlap and the most concentrated on a single
word. Each of those is a statement about a measure disagreeing with another
measure.

That suggests the question underneath the method rather than inside it. If
"diverse" is only ever "spread out under E", and different E disagree about
which corpus is spread out, then the objective is not identified: it names a
property of the embedder as much as of the corpus, and a max-min score is a
claim about a viewpoint rather than about a set of artifacts.

This is testable directly, and cheaply, on corpora already written. Six poem
arms, produced by interventions that differ in what they constrain, are scored
under five genuinely different representations:

    nomic     the semantic embedder the poem loop actually optimizes
    clip      a second semantic model, trained differently on different data
    tfidf     lexical surface, no learned semantics at all
    prosody   form -- line, metre, rhyme, punctuation
    style     function-word profile, the authorship-attribution view

If diversity is a property of the corpus, the arms rank the same way under all
five and the disagreement is noise. If it is a property of the viewpoint, the
rankings decorrelate, and the arm that wins under the objective's own embedder
is not the arm that wins elsewhere. Kendall's tau between rankings is the
statistic; a worst-case-over-representations score is the practical consequence.

Nothing here is generated. The corpora exist; only the reading changes.

Usage:
    python3 is_diversity_identified.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

from text_realization import prosodic_features        # noqa: E402
from resample import bootstrap_ci                     # noqa: E402

OLLAMA = "http://localhost:11434/api/embeddings"

# The 150 commonest English function words: the standard stylometric view, and
# deliberately the one channel that carries no content at all.
FUNCTION_WORDS = """the of and to a in that it is was i for on you he be with as by at
have are this not but had his they from she which or we an were her been has their
would there what will all if can her said who one so up out no when him my me your
me now than its into more only other some could them these two may then do does did
about after also am another any because before being between both came come did each
even every few first get go good great here how just know like little long made make
many might most much must never new now off old over own put same see should since
still such take tell too under until use very want way well went where while why
without work year yes yet young our us am upon shall thus though upon whom whose""".split()


def load_arms() -> dict[str, list[str]]:
    arms: dict[str, list[str]] = {}
    rows = [json.loads(l) for l in
            (HERE / "poem_device_arms.jsonl").read_text().splitlines() if l.strip()]
    arms["base"] = [r["text"] for r in rows if r["arm"] == "base"]
    arms["vehicle"] = [r["text"] for r in rows if r["arm"] == "device"]
    for name, fn in (("refined", "poem_refined_arm.jsonl"),
                     ("lexical", "poem_lexical_arm.jsonl"),
                     ("dev-avoid", "device_avoid_arm.jsonl"),
                     ("wordban", "poem_wordban_arm.jsonl")):
        p = HERE / fn
        if p.exists():
            arms[name] = [json.loads(l)["text"]
                          for l in p.read_text().splitlines() if l.strip()]
    return arms


# ---- representations -------------------------------------------------------

def rep_nomic(texts: list[str]) -> np.ndarray:
    out = []
    for t in texts:
        body = json.dumps({"model": "nomic-embed-text", "prompt": t[:2000]}).encode()
        req = urllib.request.Request(OLLAMA, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out.append(json.load(r)["embedding"])
    return np.array(out, dtype=float)


def rep_clip(texts: list[str]) -> np.ndarray:
    from image_steer7 import Clip
    return Clip().text(texts)


def rep_tfidf(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=4000,
                        stop_words=None)
    return np.asarray(v.fit_transform(texts).todense(), dtype=float)


def rep_prosody(texts: list[str]) -> np.ndarray:
    fs = [prosodic_features(t) for t in texts]
    names = sorted(fs[0])
    return np.array([[f[n] for n in names] for f in fs], dtype=float)


def rep_style(texts: list[str]) -> np.ndarray:
    vocab = list(dict.fromkeys(FUNCTION_WORDS))
    rows = []
    for t in texts:
        w = re.findall(r"[a-z']+", t.lower())
        n = max(len(w), 1)
        c = {x: 0 for x in vocab}
        for x in w:
            if x in c:
                c[x] += 1
        rows.append([c[x] / n for x in vocab])
    return np.array(rows, dtype=float)


REPS = {"nomic": rep_nomic, "clip": rep_clip, "tfidf": rep_tfidf,
        "prosody": rep_prosody, "style": rep_style}


# ---- scale-free diversity --------------------------------------------------

def norm_min_nn(X: np.ndarray) -> float:
    """min nearest-neighbour distance, divided by the mean pairwise distance.

    Every representation has its own units and its own concentration of
    measure, so a raw min-NN cannot be compared across them. Dividing by the
    mean pairwise distance of the same point cloud makes the score
    dimensionless and, more importantly, invariant to any global rescaling of
    the space -- which is exactly the freedom an embedding choice enjoys.
    """
    X = np.asarray(X, dtype=float)
    X = X - X.mean(0)
    s = X.std(0)
    X = X / np.where(s > 1e-12, s, 1.0)
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    iu = np.triu_indices(len(X), 1)
    mean_pair = float(D[iu].mean())
    np.fill_diagonal(D, np.inf)
    return float(D.min() / max(mean_pair, 1e-12))


def norm_p5_nn(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    X = X - X.mean(0)
    s = X.std(0)
    X = X / np.where(s > 1e-12, s, 1.0)
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    iu = np.triu_indices(len(X), 1)
    mean_pair = float(D[iu].mean())
    np.fill_diagonal(D, np.inf)
    return float(np.percentile(D.min(1), 5) / max(mean_pair, 1e-12))


def main() -> None:
    arms = load_arms()
    n = min(len(v) for v in arms.values())
    arms = {k: v[:n] for k, v in arms.items()}
    print(f"{len(arms)} arms, {n} poems each (truncated to the smallest)\n")

    names = list(arms)
    scores: dict[str, dict[str, float]] = {r: {} for r in REPS}
    p5: dict[str, dict[str, float]] = {r: {} for r in REPS}
    for rname, fn in REPS.items():
        print(f"embedding under {rname} ...", flush=True)
        for a in names:
            X = fn(arms[a])
            scores[rname][a] = norm_min_nn(X)
            p5[rname][a] = norm_p5_nn(X)

    print(f"\nnormalized min-NN (min NN / mean pairwise distance)")
    print(f"{'arm':<12}" + "".join(f"{r:>10}" for r in REPS) + f"{'worst':>10}")
    for a in names:
        row = [scores[r][a] for r in REPS]
        print(f"{a:<12}" + "".join(f"{v:>10.4f}" for v in row)
              + f"{min(row):>10.4f}")

    print(f"\nnormalized 5th-percentile NN")
    print(f"{'arm':<12}" + "".join(f"{r:>10}" for r in REPS) + f"{'worst':>10}")
    for a in names:
        row = [p5[r][a] for r in REPS]
        print(f"{a:<12}" + "".join(f"{v:>10.4f}" for v in row)
              + f"{min(row):>10.4f}")

    from scipy.stats import kendalltau
    print(f"\nKendall tau between arm rankings, representation vs representation")
    print(f"{'':<10}" + "".join(f"{r:>10}" for r in REPS))
    taus = []
    for r1 in REPS:
        cells = []
        for r2 in REPS:
            if r1 == r2:
                cells.append("     --   ")
                continue
            t, _ = kendalltau([p5[r1][a] for a in names], [p5[r2][a] for a in names])
            cells.append(f"{t:>10.2f}")
            if r1 < r2:
                taus.append(t)
        print(f"{r1:<10}" + "".join(cells))
    print(f"\nmean off-diagonal tau: {np.mean(taus):+.3f}  "
          f"(1.0 = the representations agree completely, 0 = unrelated)")

    print(f"\n{'representation':<14}{'best arm':>14}{'worst arm':>14}")
    for r in REPS:
        b = max(names, key=lambda a: p5[r][a])
        w = min(names, key=lambda a: p5[r][a])
        print(f"{r:<14}{b:>14}{w:>14}")
    best_worst = max(names, key=lambda a: min(p5[r][a] for r in REPS))
    print(f"\nbest WORST-CASE arm across all five: {best_worst}")
    json.dump({"min_nn": scores, "p5_nn": p5, "n": n},
              open(HERE / "is_diversity_identified.json", "w"), indent=2)


if __name__ == "__main__":
    main()
