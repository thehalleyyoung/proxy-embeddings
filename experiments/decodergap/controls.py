"""Controls for the near-field profile, before it is believed.

The profile compares one distance matrix (an encoder's) to another (the
decoder's) and reports that they agree more among close pairs than among far
ones. Three things other than a fact about decoders could produce that table,
and each has a control:

  A. **Comparison artifact.** Any two distance matrices over the same points
     might agree preferentially in the near field, in which case the decoder is
     doing no work. Control: profile one ENCODER against another ENCODER, with
     no decoder anywhere. This must come back flat.

  B. **Restriction of range.** Within-bin correlation is attenuated by the bin's
     spread in x, and equal-count deciles have very unequal widths. Control:
     report each decile's x-spread, and re-run on equal-WIDTH bins.

  C. **No link at all.** Control: permute the artifacts across items, breaking
     the text-to-artifact correspondence while leaving both marginal distance
     distributions exactly intact. This must come back flat.

A fourth check is reported because it needs no correlation at all and is
therefore immune to (B): the conditional mean of artifact distance given text
distance. If the law is real that curve rises steeply in the near field and
flattens; range restriction cannot manufacture that shape.

    python3 controls.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "code"))

import probe as P  # noqa: E402


def load_code():
    import domain_code as C
    rows = [json.loads(l) for l in
            (C.OUT / "executed.jsonl").read_text().splitlines()]
    seen, uniq = set(), []
    for r in rows:
        k = r["src"].strip()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return ([r["src"] for r in uniq],
            C.behavioural_distance([r["fp"] for r in uniq]))


def load_sql():
    import domain_sql as S
    rows = [json.loads(l) for l in
            (S.OUT / "executed.jsonl").read_text().splitlines()]
    seen, uniq = set(), []
    for r in rows:
        k = " ".join(r["sql"].split()).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return ([r["sql"] for r in uniq],
            S.result_distance([r["rows"] for r in uniq]))


def load_regex():
    import domain_regex as R
    rows = [json.loads(l) for l in
            (R.OUT / "executed.jsonl").read_text().splitlines()]
    seen, uniq = set(), []
    for r in rows:
        if r["pattern"] not in seen:
            seen.add(r["pattern"])
            uniq.append(r)
    return ([r["pattern"] for r in uniq],
            R.language_distance([r["hits"] for r in uniq]))


LOADERS = {"code": load_code, "sql": load_sql, "regex": load_regex}


def char_tfidf(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    V = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                        max_features=20000)
    X = V.fit_transform(texts).toarray()
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def word_tfidf(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    V = TfidfVectorizer(analyzer="word", token_pattern=r"[A-Za-z_]+",
                        ngram_range=(1, 2), min_df=2)
    X = V.fit_transform(texts).toarray()
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def summarize(name: str, TD: np.ndarray, AD: np.ndarray, n_bins: int = 10) -> dict:
    pr = P.near_field_profile(None, AD, text_dist=TD, n_bins=n_bins)
    iu = P.upper_pairs(TD.shape[0])
    td = TD[iu]
    widths = [d.hi - d.lo for d in pr.deciles]
    print(f"\n  {name}")
    print("   decile   width    corr     mean-y    (n)")
    for d, w in zip(pr.deciles, widths):
        print(f"     {d.decile:>2}    {w:.4f}  {d.corr:+.3f}    {d.mean_artifact_distance:.3f}"
              f"   {d.n_pairs}")
    print(f"   near {pr.near_corr:+.3f}  far {pr.far_corr:+.3f}  "
          f"near-far {pr.near_minus_far:+.3f}  pooled {pr.global_corr:+.3f}")
    return {"name": name, "near": pr.near_corr, "far": pr.far_corr,
            "near_minus_far": pr.near_minus_far, "pooled": pr.global_corr,
            "decile_widths": widths,
            "mean_y": [d.mean_artifact_distance for d in pr.deciles],
            "corr": [d.corr for d in pr.deciles]}


def equal_width(TD: np.ndarray, AD: np.ndarray, n_bins: int = 10) -> dict:
    """Bins of equal x-width, so every bin has the same range to correlate over."""
    iu = P.upper_pairs(TD.shape[0])
    td, ad = TD[iu], AD[iu]
    edges = np.linspace(td.min(), td.max() + 1e-12, n_bins + 1)
    rows = []
    print("\n  equal-WIDTH bins (range restriction held constant)")
    print("    bin      range            n      corr     mean-y")
    for b in range(n_bins):
        m = (td >= edges[b]) & (td < edges[b + 1])
        if m.sum() < 30 or np.ptp(ad[m]) == 0:
            rows.append(None)
            print(f"     {b+1:>2}   [{edges[b]:.3f},{edges[b+1]:.3f}]  {m.sum():>7}      --")
            continue
        r, _ = stats.pearsonr(td[m], ad[m])
        rows.append(float(r))
        print(f"     {b+1:>2}   [{edges[b]:.3f},{edges[b+1]:.3f}]  {m.sum():>7}   {r:+.3f}"
              f"     {ad[m].mean():.3f}")
    return {"equal_width_corr": rows,
            "edges": [float(e) for e in edges]}


def conditional_mean(TD: np.ndarray, AD: np.ndarray, n_bins: int = 20) -> dict:
    """E[artifact distance | text distance]. No correlation, so no attenuation."""
    iu = P.upper_pairs(TD.shape[0])
    td, ad = TD[iu], AD[iu]
    q = np.quantile(td, np.linspace(0, 1, n_bins + 1))
    q[-1] = np.nextafter(q[-1], np.inf)
    xs, ys = [], []
    for b in range(n_bins):
        m = (td >= q[b]) & (td < q[b + 1])
        if m.sum() < 5:
            continue
        xs.append(float(td[m].mean()))
        ys.append(float(ad[m].mean()))
    ys = np.array(ys)
    half = len(ys) // 2
    rise_near = float(ys[:half].max() - ys[0])
    rise_far = float(ys[-1] - ys[half])
    print("\n  conditional mean of artifact distance (immune to range restriction)")
    print("    " + "  ".join(f"{v:.3f}" for v in ys))
    print(f"    rise over the near half {rise_near:+.3f}   "
          f"over the far half {rise_far:+.3f}")
    return {"x": xs, "y": [float(v) for v in ys],
            "rise_near_half": rise_near, "rise_far_half": rise_far}


def main() -> None:
    from pipeline import embed
    which = sys.argv[1] if len(sys.argv) > 1 else "code"
    srcs, AD = LOADERS[which]()
    print(f"[{which}] {len(srcs)} distinct items")
    E_nomic = embed(srcs)
    TD_nomic = P.pairwise_cosine(E_nomic)
    TD_char = P.pairwise_cosine(char_tfidf(srcs))
    TD_word = P.pairwise_cosine(word_tfidf(srcs))

    out: dict = {"n_items": len(srcs)}
    print("\n=== A. DECODER PROFILES (the claim) ===")
    out["nomic_vs_behaviour"] = summarize("nomic  vs behaviour", TD_nomic, AD)
    out["chartfidf_vs_behaviour"] = summarize("char-tfidf vs behaviour", TD_char, AD)
    out["wordtfidf_vs_behaviour"] = summarize("word-tfidf vs behaviour", TD_word, AD)

    print("\n=== B. CONTROL: encoder vs encoder, NO decoder ===")
    print("  If these show the same near>far shape, the profile is an artifact")
    print("  of comparing two distance matrices and says nothing about decoders.")
    out["nomic_vs_char"] = summarize("nomic vs char-tfidf  [CONTROL]", TD_nomic, TD_char)
    out["nomic_vs_word"] = summarize("nomic vs word-tfidf  [CONTROL]", TD_nomic, TD_word)
    out["char_vs_word"] = summarize("char-tfidf vs word-tfidf  [CONTROL]", TD_char, TD_word)

    print("\n=== C. CONTROL: permuted artifacts ===")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(srcs))
    AD_perm = AD[np.ix_(perm, perm)]
    out["permuted"] = summarize("nomic vs PERMUTED behaviour  [CONTROL]", TD_nomic, AD_perm)

    print("\n=== D. CONTROL: range restriction ===")
    out["equal_width"] = equal_width(TD_nomic, AD)

    print("\n=== E. conditional mean (range-robust form of the claim) ===")
    out["cond_mean_decoder"] = conditional_mean(TD_nomic, AD)
    print("\n  the same curve for the encoder-vs-encoder control:")
    out["cond_mean_control"] = conditional_mean(TD_nomic, TD_char)

    (HERE / "runs" / f"controls_{which}.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote runs/controls.json")

    print("\n=== VERDICT ===")
    claim = out["nomic_vs_behaviour"]["near_minus_far"]
    ctrls = [out["nomic_vs_char"]["near_minus_far"],
             out["nomic_vs_word"]["near_minus_far"],
             out["char_vs_word"]["near_minus_far"]]
    perm_v = out["permuted"]["near_minus_far"]
    print(f"  decoder near-far          {claim:+.3f}")
    print(f"  encoder-encoder controls  {', '.join(f'{c:+.3f}' for c in ctrls)}")
    print(f"  permutation null          {perm_v:+.3f}")
    worst = max(ctrls)
    if claim > 2 * max(worst, 0.01) and abs(perm_v) < 0.1:
        print("  -> the decoder profile exceeds every decoder-free control: claim stands")
    else:
        print("  -> control reproduces the shape: the mechanism claim does NOT stand")


if __name__ == "__main__":
    main()
