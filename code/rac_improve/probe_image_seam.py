"""
Where does image conditioning lose its grip -- in the writing, or in the render?

§6.14 finds most image axes undetectable in the renders they command, and
§6.15 attributes the loss to a SEAM: the axis set is written by the model that
proposed it into a prompt, then handed to `gpt-image-1-mini`, which never saw
the axes and shares no latent space with them. That attribution has so far been
indirect -- a low correlation between text and image similarity, measured
across a corpus in which every axis co-varies with every other.

This measures it directly, and by manipulation. One base spec is held fixed,
one axis is set to each of its levels, and the SAME intervention is read twice:

  text side    CLIP embedding of the written instruction
  image side   CLIP embedding of the render made from that instruction
  pixel side   optical statistics of that render, outside CLIP entirely

Because both CLIP readings live in one space, the two numbers are directly
comparable, and the difference between them localizes the loss. A large text
effect with a small image effect puts the loss at the render boundary. Two
small effects put it in the writing step, and the seam explanation is wrong.

The corpus axes divide into two families that make this sharper than a single
average would. Seven are conceptual (*appropriation distance*, *copy and
seriality logic*); four are perceptual, added by gap expansion (*camera
viewpoint*, *shot scale*, *lighting direction*). A renderer that never saw the
axis set has no way to honour "copies outrank an absent original" but every
reason to honour "hard side-light, low angle". If the seam is real it should be
SELECTIVE, not uniform -- and which axes survive it is the practical question,
since it says what an axis set for a handed-off generator should contain.

Usage:
    python3 probe_image_seam.py [run_dir]
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

from pipeline import USAGE, chat                      # noqa: E402
import render_images                                  # noqa: E402
from render_images import _load_openai_key           # noqa: E402
from axis_realization import literal_features, eta_squared  # noqa: E402

_load_openai_key()

N_BASES = 3
N_PERM = 2000
SEED = 20260903
WORKERS = 8
OUT = HERE / "seam_probe"

# Three conceptual axes and three perceptual ones. The conceptual set includes
# `Copy and seriality logic`, which §6.14 identifies as the least realized axis
# in the corpus and the one attribution scoring ranked first.
# All eleven axes. Three per family was enough to see the split and far too
# few to test it: a family comparison over three points cannot reach p < 0.10
# however large the gap. The probe is resumable by cell, so widening it pays
# only for the axes not already on disk.
CONCEPTUAL = ["Copy and seriality logic", "Appropriation distance",
              "Self-referentiality", "Materiality disclosure",
              "Status hierarchy treatment", "Register collision",
              "Textual agency"]
PERCEPTUAL = ["[percept] Camera viewpoint and angle", "[percept] Shot scale",
              "[percept] Lighting direction and hardness",
              "[percept] Motion and temporal blur"]

# The contract block only. No avoid-block, no mined attractors, no hard bans:
# a probe holds everything except the manipulated axis fixed.
WRITE_PROMPT = """Write ONE image-generation instruction for a post-modern artwork. \
One paragraph, 30-70 words, concrete and visual.

Let these choices shape the work. Express each as something visible, never as \
abstract art-theory:
{contracts}

Name the medium, the composition, the palette, the surface, and the treatment. \
Output only the instruction."""


def final_axes(run: Path) -> dict[str, list[str]]:
    axes: dict[str, list[str]] = {}
    for line in (run / "axes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        for a in d.get("axes", []) or []:
            axes[a["name"]] = list(a["levels"])
        a = d.get("axis")
        if isinstance(a, dict) and a.get("name"):
            axes[a["name"]] = list(a["levels"])
    return axes


def render_keyed(name: str, prompt: str, out_dir: Path) -> bool:
    """`render_images.render_one` names files from an integer index; this probe
    needs cell-keyed names so the set is resumable. Same request, same retries,
    same atomic write -- only the filename differs, so the original stays
    untouched."""
    import base64, json as _json, os, time as _time, urllib.request
    path = out_dir / f"{name}.png"
    if path.exists() and path.stat().st_size > 1000:
        return True
    body = _json.dumps({"model": render_images.IMG_MODEL, "prompt": prompt[:3800],
                        "size": render_images.SIZE,
                        "quality": render_images.QUALITY, "n": 1}).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                render_images.IMG_URL, data=body,
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                payload = _json.load(r)
            b64 = payload["data"][0].get("b64_json")
            if not b64:
                raise RuntimeError("no b64_json in response")
            tmp = path.with_suffix(".png.part")
            tmp.write_bytes(base64.b64decode(b64))
            os.replace(tmp, path)
            return True
        except Exception as e:
            if attempt == 2:
                print(f"  [{name}] render failed: {str(e)[:150]}", flush=True)
                return False
            _time.sleep(2 ** attempt + 1)
    return False


def key_for(axis: str, base: int, level: str) -> str:
    """Render filename keyed by the CELL, not by position in the job list.

    Keying on list position means any change to the design -- another base, one
    more axis -- shifts every index, so `render_one`'s skip-if-exists check
    stops matching and the whole set is bought again. A cell key makes the
    probe resumable: re-running with more bases pays only for the new ones.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", axis.lower()).strip("-")[:40]
    lv = re.sub(r"[^a-z0-9]+", "-", level.lower()).strip("-")[:40]
    return f"{slug}__b{base}__{lv}"


def write_instruction(spec: dict[str, str]) -> str | None:
    contracts = "\n".join(f"  - {k}: {v}" for k, v in spec.items())
    last = None
    for attempt in range(4):
        time.sleep(random.uniform(0, 1.5) + 2.0 * attempt)
        try:
            t = chat([{"role": "user",
                       "content": WRITE_PROMPT.format(contracts=contracts)}],
                     temperature=1.0, max_tokens=600)
            if len(t.strip()) > 40:
                return t.strip()
            last = "short output"
        except Exception as e:
            last = repr(e)
    print(f"   write failed after 4 tries: {last}", flush=True)
    return None


def rho(X: np.ndarray, labels: list[str], bases: list, n_perm: int = 0):
    """Null-corrected between-level variance share, centered within base."""
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    X = (X - X.mean(0)) / np.clip(X.std(0), 1e-9, None)
    for b in set(bases):
        m = np.array([x == b for x in bases])
        X[m] -= X[m].mean(0)
    k, n = len(set(labels)), len(labels)
    if k < 2 or n < 3 * k:
        return float("nan"), float("nan")
    eta = eta_squared(X, labels)
    null = (k - 1) / (n - 1)
    r = float(np.clip((eta - null) / max(1.0 - null, 1e-9), 0.0, 1.0))
    if not n_perm:
        return r, float("nan")
    rng = np.random.default_rng(SEED)
    lab = np.array(labels)
    masks = [np.array([x == b for x in bases]) for b in set(bases)]
    hits = 0
    for _ in range(n_perm):
        q = lab.copy()
        for m in masks:
            q[m] = rng.permutation(q[m])
        if eta_squared(X, list(q)) >= eta:
            hits += 1
    return r, float((hits + 1) / (n_perm + 1))


def main(run: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    axes = final_axes(run)
    targets = [a for a in CONCEPTUAL + PERCEPTUAL if a in axes]
    print(f"{len(axes)} axes in run; probing {len(targets)}:")
    for t in targets:
        print(f"   {t}")

    jobs, meta = [], []
    for tgt in targets:
        pool = sorted(a for a in axes if a != tgt)
        for bi in range(N_BASES):
            # Seed per CELL, not from one shared stream. With a shared stream,
            # adding a base shifts every later target's draws, so previously
            # rendered cells silently change identity and the cache is void.
            r = random.Random(f"{SEED}|{tgt}|{bi}")
            base = {a: r.choice(axes[a]) for a in pool}
            keep = r.sample(pool, 6)            # load 7 including the target
            for lv in axes[tgt][:5]:
                jobs.append({tgt: lv} | {a: base[a] for a in keep})
                meta.append((tgt, bi, lv))

    cached = {}
    f_prev = HERE / "seam_instructions.jsonl"
    if f_prev.exists():
        for line in f_prev.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("instruction"):
                    cached[d.get("key") or key_for(d["axis"], d["base"], d["level"])] = d["instruction"]
    keys = [key_for(t, b, l) for t, b, l in meta]
    todo = [i for i, k in enumerate(keys) if k not in cached]
    print(f"\n{len(jobs)} cells; {len(cached)} instructions cached, "
          f"{len(todo)} to write", flush=True)
    instrs: list[str | None] = [cached.get(k) for k in keys]
    if todo:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            got = list(ex.map(write_instruction, [jobs[i] for i in todo]))
        for i, t in zip(todo, got):
            instrs[i] = t
    ok = [i for i, t in enumerate(instrs) if t]
    print(f"{len(ok)}/{len(jobs)} instructions available", flush=True)

    # Persist before rendering: renders are the expensive half and must never
    # have to be repeated because of an analysis bug.
    with open(HERE / "seam_instructions.jsonl", "w") as fh:
        for i, (tgt, bi, lv) in enumerate(meta):
            fh.write(json.dumps({"key": keys[i], "axis": tgt, "base": bi,
                                 "level": lv, "spec": jobs[i],
                                 "instruction": instrs[i]}) + "\n")

    need = [i for i in ok if not (OUT / f"{keys[i]}.png").exists()]
    print(f"{len(ok) - len(need)} renders on disk, {len(need)} to make", flush=True)
    if need:
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(lambda i: render_keyed(keys[i], instrs[i], OUT), need))
    have = [i for i in ok if (OUT / f"{keys[i]}.png").exists()]
    print(f"{len(have)}/{len(ok)} images available", flush=True)

    from image_steer7 import Clip
    clip = Clip()
    T = clip.text([instrs[i] for i in have])
    I = clip.image([OUT / f"{keys[i]}.png" for i in have])
    P = np.array([[v for _, v in sorted(literal_features(OUT / f"{keys[i]}.png").items())]
                  for i in have])

    print(f"\n{'axis':<42}{'rho text':>10}{'p':>8}{'rho img':>9}{'p':>8}"
          f"{'rho pix':>9}{'p':>8}")
    rows = []
    for tgt in targets:
        sel = [j for j, i in enumerate(have) if meta[i][0] == tgt]
        if len(sel) < 10:
            print(f"{tgt[:41]:<42}   (dropped: {len(sel)} renders)")
            continue
        lab = [meta[have[j]][2] for j in sel]
        bs = [meta[have[j]][1] for j in sel]
        rt, pt = rho(T[sel], lab, bs, N_PERM)
        ri, pi = rho(I[sel], lab, bs, N_PERM)
        rp, pp = rho(P[sel], lab, bs, N_PERM)
        fam = "percept" if tgt.startswith("[percept]") else "concept"
        rows.append({"axis": tgt, "family": fam, "n": len(sel),
                     "rho_text": rt, "p_text": pt, "rho_img": ri, "p_img": pi,
                     "rho_pix": rp, "p_pix": pp})
        print(f"{tgt[:41]:<42}{rt:>10.4f}{pt:>8.4f}{ri:>9.4f}{pi:>8.4f}"
              f"{rp:>9.4f}{pp:>8.4f}")

    for fam in ("concept", "percept"):
        g = [r for r in rows if r["family"] == fam]
        if g:
            print(f"\n{fam:<10} mean  text {np.nanmean([r['rho_text'] for r in g]):.4f}"
                  f"   image {np.nanmean([r['rho_img'] for r in g]):.4f}"
                  f"   pixels {np.nanmean([r['rho_pix'] for r in g]):.4f}")
    try:
        from scipy.stats import mannwhitneyu
        c = [r for r in rows if r["family"] == "concept"]
        q = [r for r in rows if r["family"] == "percept"]
        for ch in ("rho_text", "rho_img", "rho_pix"):
            a = [r[ch] for r in c if np.isfinite(r[ch])]
            b = [r[ch] for r in q if np.isfinite(r[ch])]
            if len(a) >= 3 and len(b) >= 3:
                u, pv = mannwhitneyu(b, a, alternative="greater")
                print(f"  {ch:<10} percept > concept: U={u:.0f} p={pv:.4f} "
                      f"({np.mean(b):.4f} vs {np.mean(a):.4f})")
    except ImportError:
        pass
    print(f"\ntext spend ${USAGE.cost_usd():.2f} (renders billed separately)")
    json.dump(rows, open(HERE / "probe_image_seam.json", "w"), indent=2)


if __name__ == "__main__":
    r = Path(sys.argv[1]) if len(sys.argv) > 1 else RESEARCH / "real" / "dalle_steer7_maxmin"
    main(r.resolve())
