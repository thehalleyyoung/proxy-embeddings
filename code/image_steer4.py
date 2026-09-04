"""
image_steer3.py plus manipulation-based axis scoring (calculus2).

Third rung of the ablation ladder:
    steer2  baseline           attribution scoring, no expansion
    steer3  + gap expansion    attribution scoring
    steer4  + causal scoring   manipulation scoring (this file)

The scorer change is a two-line swap -- `score_axes_realized` becomes
`score_axes_realized_causal` -- but it changes which axis the loop believes is
worth conditioning on, and therefore which cell gets refined. On the frozen
baseline corpus the old scorer ranked an axis first whose commanded level is
undetectable in the renders (permutation p = .674 on CLIP, .920 on pixel
statistics); the new one ranks it last. See rac_improve/LOG.md.

Each run now also writes its own axis-realization table into the summary, so
no future corpus is built on axes nobody checked.

Gap expansion is inherited from image_steer3.py; its notes follow.


RAC's recursion fires on cell saturation and splits an existing axis into
finer sub-levels -- higher resolution in directions the basis already has.
It has no way to notice a direction the basis lacks: an absent dimension
never saturates because it is never sampled. Here, twice per run, a
multimodal judge LOOKS at the last rendered images and names the
perceptual/optical dimensions the set does not vary along (viewpoint, shot
scale, lighting, depth, medium, tonal range, composition geometry); each
comes back as a new axis with concrete visible levels and joins the tree
mid-run, exactly as refine() would append a split. Everything else --
seeded axes, scoring, repulsion, budget -- is identical to image_steer.py,
so any difference against the steer2 arms is attributable to expansion.

Original mechanism notes follow.

Identical structure to lyria_steer.py, and deliberately so -- the point of the
calculus is that it is modality-agnostic once you can embed the artifact. CLIP
plays the role CLAP played: a shared text/image space, so CLIP_text(prompt) is
a free pre-render estimate of CLIP_image(render), and selection can happen over
prompts rather than over renders.

No rejection. Every image generated is kept. All the leverage is in choosing
which prompt to send, and all the learning is about which natural-language
axes actually move the picture.

Two objectives, run as separate arms over the same machinery, because they
want different things from the same budget of n renders:

  maxmin    "in n turns, make no two images resemble each other" -- pick the
            candidate whose predicted landing point is FURTHEST from the
            nearest already-rendered image. Frontier-seeking.
  coverage  "in n turns, reach as much of the space as possible" -- pick the
            candidate that brings the most currently-uncovered reference
            points within eps. Bulk-filling.

Running both on one generator, one embedder, and one budget is the cleanest
available test of the claim that these are different problems: if they were
the same, the two arms would converge to the same corpus.

The validity gate is verified, never enforced (we may not reject): each image
gets a CLIP margin between "a post-modern artwork" descriptions and generic
photo descriptions, reported as a distribution.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from axes import Axis, AxisTree, spec_text
from calculus import (coverage_levels, farthest_levels, next_condition_realized,
                      recursive_axis_order, recursive_orth_residual,
                      score_axes_coverage, score_axes_realized)
from calculus2 import (axis_realization, score_axes_coverage_causal,
                       score_axes_realized_causal)
from pipeline import HERE, USAGE, chat, parse_json
from render_images import _load_openai_key, render_one
from structural import (audit_images, audit_prompts, content_words, hue_hist,
                        jaccard, layout_sig, overused_words, tiling_score)

_load_openai_key()

TARGET = 60
WORKERS = 6
POOL = 6
QUALITY = "high"
MINE_EVERY = 20
LEDGER_SLICE = 8
LEARN_AFTER = 14
RECURSION_DEPTH = 2      # levels of the latent tree to orthogonalize within
W_RECURSIVE_ORTH = 0.6
EPS_QUANTILE = 0.10      # coverage radius, as a quantile of reference distances

ON_PROBES = [
    "a post-modern artwork",
    "a work of contemporary conceptual art",
    "an art piece using appropriation and collage",
]
OFF_PROBES = [
    "a plain photograph of an everyday object",
    "a stock photo",
    "a screenshot of a user interface",
]

WRITE_PROMPT = """Write ONE image-generation instruction for a post-modern artwork. \
One paragraph, 30-70 words, concrete and visual.

Let these choices shape the work. Express each as something visible, never as \
abstract art-theory:
{contracts}

The set so far keeps doing these things. Avoid all of them:
{avoid}

Hard constraints, non-negotiable:
{hard}

Name the medium, the composition, the palette, the surface, and the treatment. \
Output only the instruction."""

# hue-bin names for turning a measured palette attractor into words the
# generator can act on (12 bins over the PIL HSV hue circle)
HUE_NAMES = ["red", "orange", "yellow", "yellow-green", "green", "teal",
             "cyan", "azure", "blue", "violet", "magenta", "pink"]

GAP_EVERY = 20           # accepts between gap-expansion rounds (first at ~12)
GAP_FIRST = 12
GAP_SAMPLE = 12          # images shown to the gap judge per round
GAP_MAX_AXES = 4         # total perceptual axes added per run

GAP_PROMPT = """You are looking at {n} images from a corpus built to maximize \
visual diversity. Ignore what the images depict or mean. Judge only how they \
LOOK, as a photography or painting instructor would.

Name the PERCEPTUAL / OPTICAL dimensions along which this set does NOT vary. \
Consider at least: camera viewpoint and angle; shot scale (macro detail to \
distant vista); depth (flat pattern vs deep atmospheric or linear perspective); \
lighting (direction, key, hardness, time of day); medium and surface \
(photograph, ink sketch, 3D render, print, fresco...); tonal range (monochrome \
or restricted palette vs saturated); composition geometry (where visual weight \
sits, symmetry, negative space); motion and blur; figuration vs abstraction.

Pick the {k} MOST ABSENT dimensions. For each, propose an axis: a short name \
and 4-6 LEVELS. Every level must be a concrete visible treatment an image \
generator can execute, and levels within an axis must be as far apart as \
possible. Do not propose subject-matter or art-theory axes.

Return JSON only:
{{"gaps": [{{"name": "...", "missing_because": "...", "levels": ["...", "..."]}}]}}"""


def _b64_png(path: Path, side: int = 512) -> str:
    """Downscaled JPEG, not the raw PNG.

    The renders are ~3 MB each at high quality; twelve of them base64-encoded
    is a ~48 MB request body, which the API drops. The failure was silent --
    gap_expand catches broadly and returns no axes -- so the mechanism simply
    never fired. 512 px JPEG is ~40 KB and loses nothing the judge needs:
    it is being asked about viewpoint, lighting, depth and tonal range, all
    of which survive downscaling.
    """
    import base64
    import io

    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def gap_expand(tree: AxisTree, paths: list[Path], k: int = 2) -> list[Axis]:
    """Show recent renders to the multimodal judge; append the absent
    perceptual dimensions it names as new top-level axes."""
    content = [{"type": "text", "text": GAP_PROMPT.format(n=len(paths), k=k)}]
    for p in paths:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_b64_png(p)}"}})
    try:
        raw = chat([{"role": "user", "content": content}], 0.4, 1200, json_mode=True)
        obj = parse_json(raw)
    except Exception as e:
        # never silent again: a swallowed failure here is indistinguishable
        # from "the judge found no gaps", and cost a whole run once
        print(f"  gap_expand FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return []
    added = []
    for g in obj.get("gaps", [])[:k]:
        levels = [str(x) for x in g.get("levels", []) if str(x).strip()]
        name = f"[percept] {str(g.get('name', '')).strip()}"
        if len(levels) < 3 or any(a.name == name for a in tree.axes):
            continue
        ax = Axis(name=name, levels=levels)
        tree.axes.append(ax)
        tree._log({"event": "gap_expand", "axis": ax.to_json(),
                   "missing_because": str(g.get("missing_because", ""))[:300]})
        added.append(ax)
    return added


MINE_PROMPT = """These pairs of art-generation instructions produced images that an \
image embedding model judged nearly IDENTICAL, despite the instructions being written \
to differ. Say what the resulting pictures keep sharing -- palettes, recurring objects, \
compositions, textures, gestures toward "post-modern".

{pairs}

Return JSON only: {{"attractors": ["...", "..."]}} -- 4-8 concrete findings."""


class Clip:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(name).eval()
        self.proc = CLIPProcessor.from_pretrained(name)
        self.on = self.text(ON_PROBES)
        self.off = self.text(OFF_PROBES)

    @staticmethod
    def _t(out):
        import torch
        if isinstance(out, torch.Tensor):
            return out
        for a in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
            v = getattr(out, a, None)
            if v is not None:
                return v if v.dim() == 2 else v.mean(dim=1)
        raise TypeError(type(out))

    def text(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            i = self.proc(text=[t[:300] for t in texts], return_tensors="pt",
                          padding=True, truncation=True)
            f = self._t(self.model.get_text_features(**i))
        f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()

    def image(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image
        out = []
        with self.torch.no_grad():
            for i in range(0, len(paths), 8):
                imgs = [Image.open(p).convert("RGB") for p in paths[i:i + 8]]
                inp = self.proc(images=imgs, return_tensors="pt")
                f = self._t(self.model.get_image_features(**inp))
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.numpy())
        return np.vstack(out)

    def artness(self, v: np.ndarray) -> tuple[float, float]:
        v = v / max(np.linalg.norm(v), 1e-9)
        return float((self.on @ v).mean()), float((self.off @ v).mean())


def load_axes(tree: AxisTree) -> AxisTree:
    src = HERE / "real" / "dalle_ihd" / "axes.jsonl"
    if not tree.axes and src.exists():
        for line in src.read_text().splitlines():
            r = json.loads(line)
            if r["event"] == "seed":
                tree.axes = [Axis(a["name"], a["levels"]) for a in r["axes"]]
        tree._log({"event": "seed", "axes": [a.to_json() for a in tree.axes]})
    return tree


def _code_provenance() -> dict:
    """Hash the generator that produced this run.

    A loop that edits its own generators between arms cannot rely on "same
    script" meaning "same code": the token cap that invalidated a comparison in
    this work was raised four experiments earlier, in a different arm, and
    nothing in the later experiment's diff revealed it. Recording the hash at
    write time makes the question answerable afterwards instead of
    reconstructable only from memory.
    """
    import hashlib
    src = Path(__file__).resolve()
    h = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    return {"generator": src.name, "sha256_16": h,
            "mtime": __import__("datetime").datetime.fromtimestamp(
                src.stat().st_mtime).isoformat(timespec="seconds")}


def run(objective: str, target: int = TARGET, tag: str = "steer4"):
    assert objective in ("maxmin", "coverage")
    # fresh directory per tag: render_one skips files that already exist, so
    # rerunning into the old dir would silently resurrect the old images
    RUN = HERE / "real" / f"dalle_{tag}_{objective}"
    IMG = RUN / "images"
    IMG.mkdir(parents=True, exist_ok=True)
    rng = random.Random(90210 if objective == "maxmin" else 11223)
    clip = Clip()
    tree = load_axes(AxisTree(RUN / "axes.jsonl"))
    print(f"[{objective}] axes={len(tree.axes)} lattice~{tree.space_size():.1e}", flush=True)

    # reference pool for the coverage objective: a large, policy-independent
    # sample of what the generator reaches unprompted. Coverage is only
    # meaningful against a measure, and this is the honest one available.
    ref = None
    if objective == "coverage":
        rp = HERE / "real" / "dalle_naive" / f"clip_image_emb_{QUALITY}.npy"
        if not rp.exists():
            rp = HERE / "real" / "dalle_naive" / "clip_image_emb_medium.npy"
        if rp.exists():
            ref = np.load(rp)
            ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
            d = 1.0 - ref @ ref.T
            eps = float(np.quantile(d[np.triu_indices(len(ref), 1)], EPS_QUANTILE))
            print(f"[coverage] reference pool {ref.shape}, eps={eps:.4f} "
                  f"(q{EPS_QUANTILE:.2f} of its own pairwise distances)", flush=True)
        else:
            print("[coverage] no reference pool; falling back to self-coverage",
                  flush=True)
            eps = 0.25
    else:
        eps = None

    recs: list[dict] = []
    E: list[np.ndarray] = []
    attractors: list[str] = []
    sens: dict = {}
    axis_order: list[str] = []
    # literal-space state: the two quadrants CLIP cannot see
    layouts: list[np.ndarray] = []      # 16x16 luminance signatures
    tilings: list[float] = []
    hues: list[np.ndarray] = []
    bad_text_emb: list[np.ndarray] = [] # CLIP-text of prompts that produced
                                        # structurally repetitive images
    covered = np.zeros(len(ref), dtype=bool) if ref is not None else None
    gap_axes: list[str] = []
    last_gap_at = GAP_FIRST - GAP_EVERY
    renders = 0
    next_render_idx = 0        # never reused, even when a render fails
    pred, real = [], []
    t0 = time.time()
    log = open(RUN / "log.jsonl", "a")

    while len(recs) < target:
        n_slots = min(WORKERS, target - len(recs))
        chosen = []
        for _ in range(n_slots):
            pool = [tree.sample_spec(rng) for _ in range(POOL)]
            avoid = ("\n".join(f"  - {a}" for a in attractors[-LEDGER_SLICE:])
                     or "  - (nothing yet)")
            # ---- measured structural bans (generation-side ceiling fix) --
            hard_lines = []
            if len(tilings) >= 8:
                if float(np.mean([x > 0.5 for x in tilings[-24:]])) > 0.2:
                    hard_lines.append(
                        "Do NOT compose the work as a grid, tiling, array, or "
                        "rows of repeated cells, panels, stamps, or posters. "
                        "One continuous composition.")
                Lr = np.stack(layouts[-24:])
                if float((Lr @ Lr.T - np.eye(len(Lr))).max()) > 0.5:
                    hard_lines.append(
                        "The overall composition and layout must be unlike "
                        "recent works: change the framing, viewpoint, symmetry, "
                        "and where the visual weight sits.")
                Hm = np.mean(np.stack(hues[-24:]), axis=0)
                top = np.argsort(Hm)[::-1][:2]
                if float(Hm[top].sum()) > 0.5:
                    names = ", ".join(HUE_NAMES[int(i)] for i in top)
                    hard_lines.append(
                        f"The palette must NOT be dominated by {names}; commit "
                        "to a genuinely different colour world.")
            hard_lines.append(
                "No readable text, letters, numbers, or typography anywhere in "
                "the artwork, unless one of the contracts above explicitly "
                "requires text -- and then only as that contract demands.")
            hot_words = ([w for w, _ in overused_words(
                [r["prompt"] for r in recs], min_df=0.4)][:8]
                if len(recs) >= 10 else [])
            hot_words = [w for w in hot_words
                         if w not in ("create", "use", "using", "make")]
            if hot_words:
                hard_lines.append(
                    "Do not use or depict any of these overused elements: "
                    + ", ".join(hot_words) + ".")
            hard = "\n".join(f"  - {h}" for h in hard_lines)
            texts = []
            with ThreadPoolExecutor(max_workers=POOL) as ex:
                futs = {ex.submit(chat, [{"role": "user", "content": WRITE_PROMPT.format(
                    contracts=spec_text(s), avoid=avoid, hard=hard)}], 1.0, 900): s
                        for s in pool}
                for fu in as_completed(futs):
                    s = futs[fu]
                    try:
                        t = fu.result().strip().strip('"')
                        if len(t) > 40:
                            texts.append((s, t))
                    except Exception:
                        pass
            if not texts:
                continue
            Tt = clip.text([t for _, t in texts])
            # ---- literal-text repulsion (pre-render) ---------------------
            prior_words = [content_words(r["prompt"]) for r in recs]
            hot = dict(overused_words([r["prompt"] for r in recs], min_df=0.35)) \
                if len(recs) >= 10 else {}
            lit_pen = np.zeros(len(texts))
            for i, (_, t) in enumerate(texts):
                cw = content_words(t)
                jmax = max((jaccard(cw, pw) for pw in prior_words), default=0.0)
                hotshare = (sum(1 for w in cw if w in hot) / len(cw)) if cw else 0.0
                lit_pen[i] = 0.8 * max(jmax - 0.25, 0.0) + 0.5 * hotshare
            # ---- learned text->bad-structure bridge ----------------------
            if bad_text_emb:
                B = np.stack(bad_text_emb)
                lit_pen += 0.6 * np.clip((Tt @ B.T).max(axis=1) - 0.75, 0.0, None) * 4.0
            # ---- the two objectives diverge exactly here ----------------
            if objective == "maxmin":
                score = (1.0 - (Tt @ np.stack(E).T).max(axis=1)) if E else np.ones(len(texts))
            else:
                if ref is not None:
                    d = 1.0 - Tt @ ref.T                     # (cand, ref)
                    score = ((d <= eps) & ~covered[None, :]).sum(axis=1).astype(float)
                    score = score / max(len(ref), 1)
                else:
                    score = (1.0 - (Tt @ np.stack(E).T).max(axis=1)) if E else np.ones(len(texts))
            bonus = np.zeros(len(texts))
            if sens and len(recs) >= LEARN_AFTER:
                mx = max(sens.values()) or 1.0
                for i, (s, _) in enumerate(texts):
                    bonus[i] = float(np.mean([sens.get(a, 0.0) / mx for a in s]))
            # RECURSIVE orthogonalization: novel globally AND novel inside the
            # candidate's own cell of the (recursively refined) latent tree.
            # A flat residual cannot separate the k-th clone of a minority cell
            # from a genuinely new direction once the global basis has
            # truncated that cell away.
            rorth = np.zeros(len(texts))
            if len(recs) >= LEARN_AFTER and axis_order:
                A = np.stack(E); A = A / np.linalg.norm(A, axis=1, keepdims=True)
                rorth = recursive_orth_residual(
                    Tt, [s for s, _ in texts], A, [r["spec"] for r in recs],
                    axis_order, max_depth=RECURSION_DEPTH)
            k = int(np.argmax(score / max(score.max(), 1e-9)
                              + 0.5 * bonus + W_RECURSIVE_ORTH * rorth - lit_pen))
            chosen.append((*texts[k], float(score[k])))

        results = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {}
            for off, (s, p, sc) in enumerate(chosen):
                # index off a monotone render counter, NOT off len(recs).
                # render_one skips a path that already exists, so if any render
                # fails, len(recs) falls behind the number of files on disk and
                # the next batch recomputes indices that collide with images
                # already written -- render_one then "succeeds" without
                # rendering and the corpus silently ingests the same picture
                # twice, which reads as a min-distance of exactly zero.
                idx = next_render_idx + off
                futs[ex.submit(render_one, idx, p, IMG, QUALITY)] = (idx, s, p, sc)
            for fu in as_completed(futs):
                idx, s, p, sc = futs[fu]
                try:
                    ok = fu.result()
                except Exception:
                    ok = False
                results.append((idx, s, p, sc, ok))
        next_render_idx += len(chosen)
        renders += len(results)

        new_paths, new_meta = [], []
        for idx, s, p, sc, ok in sorted(results):
            f = IMG / f"{idx:05d}.png"
            if ok and f.exists():
                new_paths.append(f)
                new_meta.append((s, p, sc))
        if not new_paths:
            continue
        V = clip.image(new_paths)
        newL = [layout_sig(f) for f in new_paths]
        newT = [tiling_score(f) for f in new_paths]
        newH = [hue_hist(f) for f in new_paths]
        for (v, (s, p, sc), f), lsig, tsc, hh in zip(
                zip(V, new_meta, new_paths), newL, newT, newH):
            layout_nn = float(max((lsig @ np.stack(layouts).T).max(), -1.0)) \
                if layouts else 0.0
            layouts.append(lsig); tilings.append(tsc); hues.append(hh)
            # a render that tiled, or landed on an existing layout, marks its
            # PROMPT as structure-bad: future candidates near that prompt in
            # CLIP-text space are penalized before rendering
            if tsc > 0.5 or layout_nn > 0.55:
                bad_text_emb.append(clip.text([p])[0])
            on, off_ = clip.artness(v)
            realized = float(1.0 - max(v @ np.stack(E).T)) if E else 1.0
            rec = {"i": len(recs), "spec": s, "prompt": p, "file": f.name,
                   "predicted_score": round(sc, 5), "realized_gap": round(realized, 4),
                   "clip_on": round(on, 4), "clip_off": round(off_, 4),
                   "art_margin": round(on - off_, 4),
                   "tiling": round(tsc, 4), "layout_nn": round(layout_nn, 4)}
            recs.append(rec); E.append(v)
            pred.append(sc); real.append(realized)
            if covered is not None:
                covered |= (1.0 - ref @ v) <= eps
            log.write(json.dumps(rec) + "\n"); log.flush()

        if len(recs) >= LEARN_AFTER:
            A = np.stack(E)
            A = A / np.linalg.norm(A, axis=1, keepdims=True)
            specs_now = [r["spec"] for r in recs]
            ranked = (score_axes_realized_causal(specs_now, A) if objective == "maxmin"
                      else score_axes_coverage_causal(specs_now, A))
            sens = {s_.name: s_.promise_unspent for s_ in ranked}
            # descend the tree most-discriminative-axis-first, so the deepest
            # cells are the ones that actually separate the artifacts
            axis_order = [s_.name for s_ in ranked]

        if len(recs) and len(recs) % MINE_EVERY < WORKERS and len(E) > 10:
            A = np.stack(E)
            S = A @ A.T
            np.fill_diagonal(S, -np.inf)
            blocks, seen = [], set()
            for f in np.argsort(-S, axis=None)[:16]:
                i, j = np.unravel_index(f, S.shape)
                if i >= j or (i, j) in seen:
                    continue
                seen.add((i, j))
                blocks.append(f"--- pair (sim {S[i, j]:.3f}) ---\nA: {recs[i]['prompt'][:260]}"
                              f"\nB: {recs[j]['prompt'][:260]}")
                if len(blocks) >= 4:
                    break
            try:
                raw = chat([{"role": "user", "content": MINE_PROMPT.format(
                    pairs="\n".join(blocks))}], 0.3, 800, json_mode=True)
                new = [str(a) for a in parse_json(raw).get("attractors", [])][:8]
                if new:
                    attractors.extend(new)
                    with open(RUN / "ledger.jsonl", "a") as fh:
                        fh.write(json.dumps({"n": len(recs), "attractors": new}) + "\n")
                    print(f"  mined {len(new)} visual attractors", flush=True)
            except Exception:
                pass

        if (len(gap_axes) < GAP_MAX_AXES
                and len(recs) - last_gap_at >= GAP_EVERY and len(recs) >= GAP_FIRST):
            last_gap_at = len(recs)
            sample = [IMG / r["file"] for r in recs[-GAP_SAMPLE:]]
            added = gap_expand(tree, [p for p in sample if p.exists()],
                               k=min(2, GAP_MAX_AXES - len(gap_axes)))
            for ax in added:
                gap_axes.append(ax.name)
                print(f"  gap axis added: {ax.name} :: {ax.levels}", flush=True)

        print(f"[{objective}] kept={len(recs)}/{target} renders={renders} "
              f"${USAGE.cost_usd():.2f} {(time.time()-t0)/60:.0f}m", flush=True)

    A = np.stack(E)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    S = A @ A.T
    np.fill_diagonal(S, -np.inf)
    nn = 1.0 - S.max(axis=1)
    Ac = A - A.mean(0, keepdims=True)
    Ac = Ac / np.clip(np.linalg.norm(Ac, axis=1, keepdims=True), 1e-9, None)
    w = np.linalg.eigvalsh((Ac.T @ Ac) / len(Ac)); w = np.clip(w, 1e-12, None); w /= w.sum()
    vendi_c = float(np.exp(-(w * np.log(w)).sum()))
    from scipy.stats import spearmanr
    v = [(p, r) for p, r in zip(pred, real) if r < 1.0]
    align = float(spearmanr([a for a, _ in v], [b for _, b in v]).statistic) if len(v) > 8 else float("nan")
    img_paths = sorted(IMG.glob("*.png"))
    summary = {
        "objective": objective, "n": len(recs), "renders": renders, "rejections": 0,
        "quality": QUALITY,
        "structural_audit": audit_images(img_paths) if len(img_paths) > 3 else None,
        "prompt_audit": audit_prompts([r["prompt"] for r in recs]),
        "n_bad_structure_prompts": len(bad_text_emb),
        "clip_vendi_centered": round(vendi_c, 2),
        "clip_median_nn_dist": round(float(np.median(nn)), 4),
        "clip_min_nn_dist": round(float(nn.min()), 4),
        "coverage_of_reference": (round(float(covered.mean()), 4)
                                  if covered is not None else None),
        "coverage_eps": round(eps, 4) if eps else None,
        "art_margin_mean": round(float(np.mean([r["art_margin"] for r in recs])), 4),
        "fraction_on_concept": round(float(np.mean(
            [r["art_margin"] > 0 for r in recs])), 4),
        "pred_vs_realized_spearman": round(align, 4),
        "recursion_depth": RECURSION_DEPTH,
        "recursive_axis_order": axis_order,
        "gap_axes_added": gap_axes,
        "gap_axis_usage": {a: sum(1 for r in recs if a in r["spec"])
                           for a in gap_axes},
        "scoring": "manipulation (calculus2.score_axes_*_causal)",
        "code_provenance": _code_provenance(),
        # every run self-audits: which axes the generator actually obeyed,
        # bias-corrected against the exchangeable-label null
        "axis_realization": {k: round(v, 4) for k, v in sorted(
            axis_realization([r["spec"] for r in recs], A).items(),
            key=lambda kv: -kv[1])},
        "calculus_ranking": [s_.to_json() for s_ in (
            score_axes_realized_causal([r["spec"] for r in recs], A)
            if objective == "maxmin"
            else score_axes_coverage_causal([r["spec"] for r in recs], A))],
        "openrouter_cost_usd": round(USAGE.cost_usd(), 4),
        "wall_clock_min": round((time.time() - t0) / 60, 1),
    }
    np.save(RUN / "clip_emb.npy", A)
    json.dump(summary, open(RUN / "run_summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "calculus_ranking"}, indent=2))
    log.close()


def _cov_args(recs, A):
    from calculus import realized_level_usage, realized_level_vectors
    return (realized_level_vectors([r["spec"] for r in recs], A), A,
            realized_level_usage([r["spec"] for r in recs]))


if __name__ == "__main__":
    obj = sys.argv[1] if len(sys.argv) > 1 else "maxmin"
    run(obj, int(sys.argv[2]) if len(sys.argv) > 2 else TARGET,
        sys.argv[3] if len(sys.argv) > 3 else "steer4")
