"""
Does an axis actually show up in the images?

RAC conditions a prompt on a spec (one level per axis) and then trusts that the
render obeys. Nothing in the pipeline checks that. An axis that the image model
ignores is worse than useless: it consumes a slot in the spec, it is scored by
the calculus as if it were doing work, and its "promise" is spent on nothing.
This is the measurement that tells a real dimension from a decorative one.

Three independent tests per axis, all on the RENDERED image, none on the prompt:

  1. IDENTIFICATION (CLIP, cross-modal). Embed each of the axis's level strings
     with CLIP-text. For an image whose spec said level L, is CLIP_text(L) the
     nearest level? Chance is 1/n_levels; we report accuracy, lift over chance,
     and an exact-ish permutation p-value. This asks: is the commanded level
     legible in the picture?

  2. SEPARATION (CLIP, image-only). Group image embeddings by commanded level
     and compute eta^2 (between-level variance over total). This asks a weaker
     but assumption-free question: do the levels produce systematically
     different pictures at all, whether or not they match the words?

  3. LITERAL SEPARATION (pixels, no CLIP). The same eta^2 on measured optical
     features -- luminance, contrast, saturation, colourfulness, hue entropy,
     edge density, tiling, layout. CLIP is the metric RAC optimizes, so CLIP
     agreeing with RAC is partly circular; pixel statistics are outside that
     loop. An axis that moves lighting or tonal range MUST show up here.

Every eta^2 gets a label-permutation p-value, because with ~60 items and 5
levels a moderate eta^2 arises by chance often enough to fool you.

Usage:
    python3 axis_realization.py ../real/dalle_steer3_maxmin [more run dirs...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))

N_PERM = 2000
RNG = np.random.default_rng(20260901)


# ---------------------------------------------------------------- features
def literal_features(path: Path) -> dict[str, float]:
    """Optical statistics measured off the pixels, outside CLIP's opinion."""
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((256, 256))
    a = np.asarray(im, dtype=np.float64) / 255.0
    lum = a @ np.array([0.2126, 0.7152, 0.0722])
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.clip(mx, 1e-9, None), 0.0)
    # Hasler-Susstrunk colourfulness
    rg = a[:, :, 0] - a[:, :, 1]
    yb = 0.5 * (a[:, :, 0] + a[:, :, 1]) - a[:, :, 2]
    colourful = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
    # hue entropy over 12 bins, weighted by saturation (grey pixels have no hue)
    import colorsys
    hsv = np.asarray(im.convert("HSV"), dtype=np.float64) / 255.0
    hh, _ = np.histogram(hsv[:, :, 0], bins=12, range=(0, 1),
                         weights=hsv[:, :, 1])
    hh = hh / max(hh.sum(), 1e-9)
    hue_entropy = float(-(hh * np.log(np.clip(hh, 1e-12, None))).sum())
    # edge density and directionality: proxies for depth-of-field and geometry
    gx = np.abs(np.diff(lum, axis=1)).mean()
    gy = np.abs(np.diff(lum, axis=0)).mean()
    # radial luminance gradient: centred subject vs distributed weight
    yy, xx = np.mgrid[0:256, 0:256]
    r = np.hypot(yy - 127.5, xx - 127.5) / 180.0
    centre = float(lum[r < 0.35].mean() - lum[r > 0.75].mean())
    # tonal range: how much of the histogram is actually used
    q = np.quantile(lum, [0.05, 0.95])
    return {
        "luminance": float(lum.mean()),
        "contrast": float(lum.std()),
        "tonal_range": float(q[1] - q[0]),
        "saturation": float(sat.mean()),
        "colourfulness": colourful,
        "hue_entropy": hue_entropy,
        "edge_density": float(0.5 * (gx + gy)),
        "edge_anisotropy": float(abs(gx - gy) / max(gx + gy, 1e-9)),
        "centre_weight": centre,
    }


# ---------------------------------------------------------------- statistics
def eta_squared(X: np.ndarray, labels: list[str]) -> float:
    """Between-group share of total variance, for vectors or scalars."""
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    uniq = sorted(set(labels))
    if len(uniq) < 2:
        return float("nan")
    lab = np.array(labels)
    grand = X.mean(axis=0, keepdims=True)
    ss_tot = float(((X - grand) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    ss_between = 0.0
    for u in uniq:
        g = X[lab == u]
        if len(g) == 0:
            continue
        ss_between += len(g) * float(((g.mean(axis=0, keepdims=True) - grand) ** 2).sum())
    return ss_between / ss_tot


def perm_p(stat_fn, X, labels: list[str], observed: float, n: int = N_PERM) -> float:
    """One-sided permutation p-value on shuffled level labels."""
    if not np.isfinite(observed):
        return float("nan")
    lab = np.array(labels)
    hits = 0
    for _ in range(n):
        hits += stat_fn(X, list(RNG.permutation(lab))) >= observed
    return float((hits + 1) / (n + 1))


def identification(img_emb: np.ndarray, level_emb: np.ndarray,
                   labels: list[str], levels: list[str]) -> tuple[float, float]:
    """Top-1 CLIP level-identification accuracy, and its permutation p."""
    sim = img_emb @ level_emb.T                      # (items, levels)
    pred = [levels[i] for i in sim.argmax(axis=1)]
    acc = float(np.mean([p == t for p, t in zip(pred, labels)]))
    hits = 0
    lab = np.array(labels)
    for _ in range(N_PERM):
        sh = RNG.permutation(lab)
        hits += float(np.mean([p == t for p, t in zip(pred, sh)])) >= acc
    return acc, float((hits + 1) / (N_PERM + 1))


# ---------------------------------------------------------------- clip
def clip_encoder():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).eval()
    proc = CLIPProcessor.from_pretrained(name)

    def _t(out):
        if isinstance(out, torch.Tensor):
            return out
        for a in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
            v = getattr(out, a, None)
            if v is not None:
                return v if v.dim() == 2 else v.mean(dim=1)
        raise TypeError(type(out))

    def text(texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            i = proc(text=[t[:300] for t in texts], return_tensors="pt",
                     padding=True, truncation=True)
            f = _t(model.get_text_features(**i))
        return (f / f.norm(dim=-1, keepdim=True)).numpy()

    def image(paths: list[Path]) -> np.ndarray:
        from PIL import Image
        out = []
        with torch.no_grad():
            for i in range(0, len(paths), 8):
                imgs = [Image.open(p).convert("RGB") for p in paths[i:i + 8]]
                f = _t(model.get_image_features(**proc(images=imgs, return_tensors="pt")))
                out.append((f / f.norm(dim=-1, keepdim=True)).numpy())
        return np.vstack(out)

    return text, image


# ---------------------------------------------------------------- driver
def audit_run(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    recs = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines() if l.strip()]
    img_dir = run_dir / "images"
    recs = [r for r in recs if (img_dir / r["file"]).exists()]
    if len(recs) < 8:
        raise SystemExit(f"{run_dir.name}: only {len(recs)} usable records")

    # which axes were added mid-run by gap expansion, and which were seeded
    gap_axes, seed_axes = [], []
    ax_path = run_dir / "axes.jsonl"
    if ax_path.exists():
        for line in ax_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("event") == "seed":
                seed_axes = [a["name"] for a in r["axes"]]
            elif r.get("event") in ("gap_expand", "refine"):
                gap_axes.append(r["axis"]["name"])

    paths = [img_dir / r["file"] for r in recs]
    text_enc, image_enc = clip_encoder()
    E = image_enc(paths)
    F = np.array([[v for _, v in sorted(literal_features(p).items())] for p in paths])
    fnames = sorted(literal_features(paths[0]).keys())
    # standardize literal features so eta^2 is not dominated by scale
    F = (F - F.mean(0)) / np.clip(F.std(0), 1e-9, None)

    axes_seen: dict[str, list[str]] = {}
    for r in recs:
        for k, v in r["spec"].items():
            axes_seen.setdefault(k, []).append(v)

    out = {"run": run_dir.name, "n_items": len(recs), "axes": []}
    for axis, _ in sorted(axes_seen.items()):
        idx = [i for i, r in enumerate(recs) if axis in r["spec"]]
        labels = [recs[i]["spec"][axis] for i in idx]
        levels = sorted(set(labels))
        if len(idx) < 8 or len(levels) < 2:
            continue
        Ei, Fi = E[idx], F[idx]
        acc, acc_p = identification(Ei, text_enc(levels), labels, levels)
        e_clip = eta_squared(Ei, labels)
        e_lit = eta_squared(Fi, labels)
        per_feat = {}
        for j, fn in enumerate(fnames):
            e = eta_squared(Fi[:, j], labels)
            per_feat[fn] = round(float(e), 3)
        kind = ("gap" if axis in gap_axes else "seed" if axis in seed_axes else "other")
        out["axes"].append({
            "axis": axis, "kind": kind, "n": len(idx), "levels": len(levels),
            "clip_id_acc": round(acc, 3), "chance": round(1 / len(levels), 3),
            "clip_id_lift": round(acc * len(levels), 2), "clip_id_p": round(acc_p, 4),
            "clip_eta2": round(float(e_clip), 3),
            "clip_eta2_p": round(perm_p(eta_squared, Ei, labels, e_clip), 4),
            "literal_eta2": round(float(e_lit), 3),
            "literal_eta2_p": round(perm_p(eta_squared, Fi, labels, e_lit), 4),
            "top_literal_features": dict(sorted(per_feat.items(),
                                                key=lambda kv: -kv[1])[:3]),
        })
    out["axes"].sort(key=lambda a: -a["clip_eta2"])
    return out


def report(res: dict) -> str:
    lines = [f"\n=== {res['run']}  (n={res['n_items']}) ===",
             f"{'axis':<44} {'kind':<5} {'lvl':>3} {'CLIP-id':>8} {'p':>7} "
             f"{'clipη²':>7} {'p':>7} {'litη²':>7} {'p':>7}  top literal"]
    for a in res["axes"]:
        star = "*" if (a["clip_id_p"] < 0.05 or a["literal_eta2_p"] < 0.05) else " "
        lines.append(
            f"{star}{a['axis'][:43]:<43} {a['kind']:<5} {a['levels']:>3} "
            f"{a['clip_id_acc']:>4.2f}/{a['chance']:.2f} {a['clip_id_p']:>7.3f} "
            f"{a['clip_eta2']:>7.3f} {a['clip_eta2_p']:>7.3f} "
            f"{a['literal_eta2']:>7.3f} {a['literal_eta2_p']:>7.3f}  "
            + ", ".join(f"{k} {v}" for k, v in a["top_literal_features"].items()))
    for kind in ("gap", "seed"):
        sub = [a for a in res["axes"] if a["kind"] == kind]
        if sub:
            lines.append(
                f"  mean over {kind} axes: CLIP-id lift "
                f"{np.mean([a['clip_id_lift'] for a in sub]):.2f}x, "
                f"clip eta2 {np.mean([a['clip_eta2'] for a in sub]):.3f}, "
                f"literal eta2 {np.mean([a['literal_eta2'] for a in sub]):.3f} "
                f"({len(sub)} axes)")
    lines.append("  * = commanded level is detectable in the render (p<0.05 on "
                 "CLIP identification or pixel statistics)")
    return "\n".join(lines)


if __name__ == "__main__":
    dirs = [Path(d) for d in sys.argv[1:]] or [RESEARCH / "real" / "dalle_steer3_maxmin"]
    allres = []
    for d in dirs:
        r = audit_run(d)
        allres.append(r)
        print(report(r), flush=True)
    json.dump(allres, open(HERE / "axis_realization.json", "w"), indent=2)
    print(f"\nwrote {HERE / 'axis_realization.json'}")
