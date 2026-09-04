"""
Literal-space signatures for images and prompts: the two quadrants the
embedding stack cannot see.

The steered image runs looked diverse to their own instruments and are
visibly repetitive to a person: tiled grids of one repeated cell, prominent
typography, one vintage-poster palette, the same idea recolored. Each of
those lives precisely where our measurements were not looking:

                     latent (semantic)          literal (surface)
    text             nomic / CLIP-text          n-gram & content-word overlap
    vision           CLIP image                 layout, tiling, palette

CLIP is trained to align images with captions, so it encodes what is
DEPICTED and is close to invariant to composition and palette -- two images
of different subjects laid out as the same 3x3 grid in the same beige/red
palette are far apart in CLIP and near-identical to a human glance. The
signatures here are deliberately dumb, local, and non-semantic, because
"dumb and local" is exactly the level at which the repetition happened.

  layout_sig    16x16 grayscale luminance map, per-image normalized.
                Cosine similarity between layout signatures is high for
                images with the same composition regardless of subject or
                color -- the "same idea in a different color" detector.
  tiling_score  peak spatial autocorrelation of the thumbnail away from the
                origin. A grid of repeated cells autocorrelates with itself
                at the cell pitch; a non-repeating composition does not.
  hue_hist      saturation-weighted hue histogram. Catches the shared-palette
                attractor directly.
  content_words prompt-side literal signature: lowercased alphabetic tokens
                minus stopwords. High Jaccard overlap between prompts means
                the same nouns are being redrawn, whatever the embeddings say.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_STOP = set("""a an and are as at be but by for from has have if in into is it its of on or
that the their this to was were will with without not no one two three each every very
more most other some such only own same so than too can could should would may might
must do does did done being been over under between about against""".split())

_WORD = re.compile(r"[a-z]{3,}")


# --------------------------------------------------------------- vision side
def _thumb(path: Path, size: int = 64) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("L").resize((size, size))
    a = np.asarray(im, dtype=np.float64)
    return a


def layout_sig(path: Path, grid: int = 16) -> np.ndarray:
    """Per-image-normalized coarse luminance map, flattened and unit-normed."""
    a = _thumb(path, 64)
    h = a.reshape(grid, 64 // grid, grid, 64 // grid).mean(axis=(1, 3))
    h = h - h.mean()
    n = np.linalg.norm(h)
    return (h / n).ravel() if n > 1e-9 else h.ravel()


def tiling_score(path: Path) -> float:
    """Max autocorrelation of the luminance thumbnail at a nonzero shift.

    Shifts from 1/8 to 1/2 of the image in each axis: a 2x2..8x8 tiling has a
    strong peak in that band; a photographic or single-composition image does
    not. Returns a value in [0, 1]."""
    a = _thumb(path, 64)
    a = a - a.mean()
    denom = float((a * a).sum())
    if denom < 1e-9:
        return 0.0
    best = 0.0
    for axis in (0, 1):
        for shift in range(8, 33, 4):
            r = float((a * np.roll(a, shift, axis=axis)).sum()) / denom
            best = max(best, r)
    return best


def hue_hist(path: Path, bins: int = 12) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("HSV").resize((64, 64))
    a = np.asarray(im, dtype=np.float64)
    hgt, _, _ = np.histogram(a[..., 0].ravel(), bins=bins, range=(0, 255),
                             weights=a[..., 1].ravel())[0], None, None
    s = hgt.sum()
    return hgt / s if s > 1e-9 else hgt


# ----------------------------------------------------------------- text side
def content_words(text: str) -> set:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overused_words(prompts: list[str], min_df: float = 0.35,
                   top: int = 20) -> list[tuple[str, float]]:
    """Content words appearing in more than `min_df` of prompts -- the literal
    text attractors, computed rather than guessed."""
    from collections import Counter
    n = max(len(prompts), 1)
    df = Counter()
    for p in prompts:
        df.update(content_words(p))
    hot = [(w, c / n) for w, c in df.items() if c / n >= min_df]
    return sorted(hot, key=lambda x: -x[1])[:top]


# ------------------------------------------------------------- corpus audit
def audit_images(paths: list[Path]) -> dict:
    """The literal-vision repetition profile of a rendered set."""
    L = np.stack([layout_sig(p) for p in paths])
    T = np.array([tiling_score(p) for p in paths])
    H = np.stack([hue_hist(p) for p in paths])
    SL = L @ L.T
    np.fill_diagonal(SL, -np.inf)
    layout_nn = SL.max(axis=1)
    DH = np.abs(H[:, None, :] - H[None, :, :]).sum(axis=-1) / 2.0
    np.fill_diagonal(DH, np.inf)
    hue_nn = DH.min(axis=1)
    return {
        "n": len(paths),
        "layout_nn_cos_mean": round(float(layout_nn.mean()), 4),
        "layout_nn_cos_max": round(float(layout_nn.max()), 4),
        "frac_layout_nn_over_0.5": round(float((layout_nn > 0.5).mean()), 4),
        "tiling_mean": round(float(T.mean()), 4),
        "frac_tiled_over_0.5": round(float((T > 0.5).mean()), 4),
        "hue_nn_dist_mean": round(float(hue_nn.mean()), 4),
        "frac_same_palette_under_0.15": round(float((hue_nn < 0.15).mean()), 4),
    }


def audit_prompts(prompts: list[str]) -> dict:
    sets = [content_words(p) for p in prompts]
    n = len(sets)
    jj = []
    for i in range(n):
        best = 0.0
        for j in range(n):
            if i != j:
                best = max(best, jaccard(sets[i], sets[j]))
        jj.append(best)
    hot = overused_words(prompts)
    return {
        "prompt_nn_jaccard_mean": round(float(np.mean(jj)), 4),
        "prompt_nn_jaccard_max": round(float(np.max(jj)), 4),
        "overused_words": [(w, round(f, 3)) for w, f in hot[:12]],
    }


if __name__ == "__main__":
    import json
    import sys
    HERE = Path(__file__).resolve().parent
    for arm in (sys.argv[1:] or ["maxmin", "coverage"]):
        d = HERE / "real" / f"dalle_steer_{arm}"
        imgs = sorted((d / "images").glob("*.png"))
        prompts = [json.loads(l)["prompt"] for l in (d / "log.jsonl").read_text().splitlines()
                   if l.strip()]
        print(f"=== {arm} ===")
        print(json.dumps(audit_images(imgs), indent=2))
        print(json.dumps(audit_prompts(prompts), indent=2))
