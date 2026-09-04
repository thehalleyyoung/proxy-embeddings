"""
The axis-realization audit, ported from images to text.

`axis_realization.py` asks whether an axis's commanded level is detectable in a
rendered image, using CLIP for the semantic channel and pixel statistics for a
channel outside the embedding the method optimizes. Text needs the same two
channels and the same null correction; only the second one has to be rebuilt,
because a poem's "pixel statistics" are its PROSODY.

That distinction matters more here than it did for images. The embedding
(nomic-embed-text) is a semantic model: it reads what a poem is about and
roughly how it is said, and it is almost blind to line length, stanza shape,
rhyme, metre and punctuation -- exactly the formal dimensions a poet would
name first. An axis can therefore look realized in embedding space while
changing nothing about the poem AS VERSE, and an axis that restructures the
line can look inert. The prosodic channel below is the check on that.

Features are deliberately form-first, not content:
  shape      line and stanza counts, line-length mean and variability
  metre      syllables per line, and the regularity of that count
  sound      end-rhyme density, alliteration rate
  syntax     sentence length, punctuation and dash rates, question rate
  diction    type-token ratio, mean word length, latinate-suffix rate
  deixis     first / second / third person rates, tense markers

Usage:
    python3 text_realization.py ../live/poems [more corpora...]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))

N_PERM = 2000
RNG = np.random.default_rng(20260901)

VOWELS = "aeiouy"
LATINATE = ("tion", "sion", "ment", "ance", "ence", "ity", "ous", "ive", "al")


def syllables(word: str) -> int:
    """Vowel-group count with a silent-e correction. Crude, but consistent --
    we need a metre PROXY that ranks lines, not a pronouncing dictionary."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(rf"[{VOWELS}]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(n, 1)


def rhyme_key(word: str) -> str:
    """Last vowel group onward: a cheap stand-in for the rhyming part."""
    w = re.sub(r"[^a-z]", "", word.lower())
    m = list(re.finditer(rf"[{VOWELS}]+", w))
    return w[m[-1].start():] if m else w


def prosodic_features(text: str) -> dict[str, float]:
    lines = [l for l in text.strip().splitlines()]
    nonblank = [l.strip() for l in lines if l.strip()]
    if not nonblank:
        return {}
    stanzas = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    words_per_line = [re.findall(r"[A-Za-z']+", l) for l in nonblank]
    syl_per_line = [sum(syllables(w) for w in ws) for ws in words_per_line]
    all_words = [w.lower() for ws in words_per_line for w in ws]
    n_words = max(len(all_words), 1)

    ends = [ws[-1] for ws in words_per_line if ws]
    keys = [rhyme_key(w) for w in ends]
    rep = sum(c - 1 for c in Counter(keys).values() if c > 1)
    rhyme_density = rep / max(len(keys), 1)

    firsts = [ws[0][0].lower() for ws in words_per_line if ws and ws[0]]
    allit = sum(c - 1 for c in Counter(firsts).values() if c > 1) / max(len(firsts), 1)

    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    sent_len = np.mean([len(re.findall(r"[A-Za-z']+", s)) for s in sentences]) \
        if sentences else 0.0

    def rate(pat: str) -> float:
        return len(re.findall(pat, text)) / max(len(text) / 100.0, 1e-9)

    p1 = sum(1 for w in all_words if w in ("i", "me", "my", "mine", "we", "us", "our"))
    p2 = sum(1 for w in all_words if w in ("you", "your", "yours", "thee", "thy", "thou"))
    p3 = sum(1 for w in all_words if w in ("he", "she", "they", "him", "her", "them",
                                           "his", "their", "it", "its"))
    past = sum(1 for w in all_words if w.endswith("ed"))
    modal = sum(1 for w in all_words if w in ("would", "could", "might", "will",
                                              "shall", "may", "must"))
    caps = sum(1 for l in nonblank if l[:1].isupper()) / len(nonblank)

    return {
        "n_lines": float(len(nonblank)),
        "n_stanzas": float(len(stanzas)),
        "line_chars_mean": float(np.mean([len(l) for l in nonblank])),
        "line_chars_sd": float(np.std([len(l) for l in nonblank])),
        "syl_per_line_mean": float(np.mean(syl_per_line)) if syl_per_line else 0.0,
        # LOW sd = metrical regularity; this is the single most diagnostic
        # feature separating formal verse from free verse
        "syl_per_line_sd": float(np.std(syl_per_line)) if syl_per_line else 0.0,
        "rhyme_density": float(rhyme_density),
        "alliteration": float(allit),
        "sentence_len": float(sent_len),
        "comma_rate": rate(r","),
        "dash_rate": rate(r"[-–—]"),
        "question_rate": rate(r"\?"),
        "colon_semicolon_rate": rate(r"[;:]"),
        "type_token": len(set(all_words)) / n_words,
        "word_len": float(np.mean([len(w) for w in all_words])) if all_words else 0.0,
        "latinate": sum(1 for w in all_words if w.endswith(LATINATE)) / n_words,
        "first_person": p1 / n_words,
        "second_person": p2 / n_words,
        "third_person": p3 / n_words,
        "past_marker": past / n_words,
        "modal": modal / n_words,
        "line_initial_caps": float(caps),
    }


# ------------------------------------------------------------------ stats
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


def audit_corpus(corpus_dir: Path) -> dict:
    path = corpus_dir / "corpus.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("spec") and (r.get("text") or "").strip()]
    if len(recs) < 8:
        raise SystemExit(f"{corpus_dir.name}: only {len(recs)} usable records")

    texts = [r["text"] for r in recs]
    feats = [prosodic_features(t) for t in texts]
    fnames = sorted(feats[0].keys())
    F = np.array([[f.get(k, 0.0) for k in fnames] for f in feats])
    F = (F - F.mean(0)) / np.clip(F.std(0), 1e-9, None)

    if "emb" in recs[0] and recs[0]["emb"]:
        E = np.array([r["emb"] for r in recs], dtype=float)
    else:
        from pipeline import embed
        E = embed(texts)
    E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)

    axes_seen: dict[str, list[str]] = {}
    for r in recs:
        for k, v in r["spec"].items():
            axes_seen.setdefault(k, []).append(str(v))

    out = {"corpus": corpus_dir.name, "n_items": len(recs), "axes": []}
    for axis in sorted(axes_seen):
        idx = [i for i, r in enumerate(recs) if axis in r["spec"]]
        labels = [str(recs[i]["spec"][axis]) for i in idx]
        if len(idx) < 8 or len(set(labels)) < 2:
            continue
        Ei, Fi = E[idx], F[idx]
        e_sem, e_pro = eta_squared(Ei, labels), eta_squared(Fi, labels)
        per_feat = {fn: round(float(eta_squared(Fi[:, j], labels)), 3)
                    for j, fn in enumerate(fnames)}
        out["axes"].append({
            "axis": axis, "n": len(idx), "levels": len(set(labels)),
            "semantic_eta2": round(float(e_sem), 3),
            "semantic_p": round(perm_p(Ei, labels, e_sem), 4),
            "prosodic_eta2": round(float(e_pro), 3),
            "prosodic_p": round(perm_p(Fi, labels, e_pro), 4),
            "top_prosodic": dict(sorted(per_feat.items(), key=lambda kv: -kv[1])[:3]),
        })
    out["axes"].sort(key=lambda a: -a["semantic_eta2"])

    # corpus-level uniformity: what the whole set never varies
    A = np.array([[f.get(k, 0.0) for k in fnames] for f in feats])
    out["corpus_form"] = {k: {"mean": round(float(A[:, j].mean()), 3),
                              "cv": round(float(A[:, j].std() /
                                                max(abs(A[:, j].mean()), 1e-9)), 3)}
                          for j, k in enumerate(fnames)}
    return out


def report(res: dict) -> str:
    L = [f"\n=== {res['corpus']}  (n={res['n_items']}) ===",
         f"{'axis':<34} {'lvl':>3} {'sem η²':>7} {'p':>6} {'pros η²':>8} {'p':>6}"
         f"  top prosodic features"]
    for a in res["axes"]:
        star = "*" if (a["semantic_p"] < 0.05 or a["prosodic_p"] < 0.05) else " "
        L.append(f"{star}{a['axis'][:33]:<33} {a['levels']:>3} "
                 f"{a['semantic_eta2']:>7.3f} {a['semantic_p']:>6.3f} "
                 f"{a['prosodic_eta2']:>8.3f} {a['prosodic_p']:>6.3f}  "
                 + ", ".join(f"{k} {v}" for k, v in a["top_prosodic"].items()))
    cf = res["corpus_form"]
    flat = sorted(cf.items(), key=lambda kv: abs(kv[1]["cv"]))[:8]
    L.append("\n  form dimensions the corpus barely varies (lowest coefficient "
             "of variation):")
    for k, v in flat:
        L.append(f"    {k:<24} mean {v['mean']:>8.3f}   cv {v['cv']:>6.3f}")
    L.append("  * = commanded level detectable (p<0.05, semantic or prosodic)")
    return "\n".join(L)


if __name__ == "__main__":
    dirs = [Path(d) for d in sys.argv[1:]] or [RESEARCH / "live" / "poems"]
    out = []
    for d in dirs:
        r = audit_corpus(d.resolve())
        out.append(r)
        print(report(r), flush=True)
    json.dump(out, open(HERE / "text_realization.json", "w"), indent=2)
    print(f"\nwrote {HERE / 'text_realization.json'}")
