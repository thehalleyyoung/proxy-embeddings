"""Domain: generated SVG, decoded through its XML into what is actually drawn.

SVG is the case the study was missing: a widely used generation target that is a
*file format*, parsed by a real XML parser rather than by anything we wrote, and
whose artifact is visual. It is also the honest version of the image work in the
companion paper, where the decoder is a diffusion model with no exposed seed and
therefore no artifact-space number can be repeated. An SVG document decodes the
same way every time.

The artifact is deliberately **not** a raster. Rasterizing would add a renderer
we do not control and a dependency, and the interesting content survives without
it: the decoded document is reduced to the set of *marks* it makes — for each
drawable element, its tag, where its centre falls on a coarse grid, how large it
is, and its fill hue, all quantized. Two documents are close when they put
similar marks in similar places, which is a property of the drawing rather than
of the markup that produced it.

Distance is Jaccard between mark sets; coverage is the number of distinct marks
a set of documents makes between them.

    python3 domain_svg.py generate 700
    python3 domain_svg.py execute
    python3 domain_svg.py probe
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "runs" / "svg"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

VIEW = 100.0          # documents are normalized into a 100x100 box
GRID = 6              # position quantization
SIZE_BANDS = (0.01, 0.05, 0.15, 0.4)
HUES = 8

PROMPT = (
    "Write ONE self-contained SVG document, at most 25 elements, with "
    'viewBox="0 0 100 100". Use only these elements: rect, circle, ellipse, '
    "line, polygon, polyline, path, g. Give shapes explicit fill or stroke "
    "colours as hex. No scripts, no external references, no text elements, no "
    "animation, no CSS classes. Draw something specific and interesting. "
    "Reply with the SVG only, in a ```xml code block, no commentary."
)

_BAD = re.compile(r"<\s*(script|foreignObject|image|use|iframe)\b|xlink:href|"
                  r"href\s*=|url\s*\(|javascript:", re.I)


def is_safe(doc: str) -> tuple[bool, str]:
    if _BAD.search(doc):
        return False, "external or scripted content"
    if "<svg" not in doc.lower():
        return False, "no svg root"
    if len(doc) > 20000:
        return False, "too large"
    return True, ""


def extract(text: str) -> str | None:
    text = text or ""
    for p in text.split("```")[1:]:
        for tag in ("xml", "svg", "html"):
            if p.lower().startswith(tag):
                p = p[len(tag):]
                break
        p = p.strip()
        if "<svg" in p.lower():
            return p
    i = text.lower().find("<svg")
    if i >= 0:
        j = text.lower().rfind("</svg>")
        return text[i:j + 6] if j > i else None
    return None


def generate(n: int) -> None:
    from pipeline import chat

    def one(i: int) -> dict:
        try:
            t = chat([{"role": "user", "content": PROMPT}],
                     temperature=1.0, max_tokens=1800)
            if not (t or "").strip():
                t = chat([{"role": "user", "content": PROMPT}],
                         temperature=1.0, max_tokens=3000)
        except Exception as e:
            return {"i": i, "error": str(e)[:200]}
        doc = extract(t)
        if not doc:
            return {"i": i, "error": "no svg"}
        ok, why = is_safe(doc)
        return {"i": i, "svg": doc, "safe": ok, "reject": why}

    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(one, range(n)))
    with (OUT / "corpus.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"generated {len(rows)}, safe {sum(1 for r in rows if r.get('safe'))}")


# ------------------------------------------------------------------ decode

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_DRAW = {"rect", "circle", "ellipse", "line", "polygon", "polyline", "path"}


def _hue_band(colour: str | None) -> int:
    if not colour:
        return HUES          # "no fill" is its own band
    c = colour.strip().lower()
    m = re.fullmatch(r"#([0-9a-f]{3}|[0-9a-f]{6})", c)
    if not m:
        named = {"black": 0, "white": 0, "red": 0, "orange": 30, "yellow": 60,
                 "green": 120, "cyan": 180, "blue": 240, "purple": 280,
                 "magenta": 300, "grey": 0, "gray": 0, "none": None}
        if c in named:
            return HUES if named[c] is None else int(named[c] / 360 * HUES) % HUES
        return HUES
    h = m.group(1)
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 0.08:
        return HUES          # achromatic
    if mx == r:
        hue = (60 * ((g - b) / (mx - mn))) % 360
    elif mx == g:
        hue = 60 * ((b - r) / (mx - mn)) + 120
    else:
        hue = 60 * ((r - g) / (mx - mn)) + 240
    return int(hue / 360 * HUES) % HUES


def _extent(el, tag: str) -> tuple[float, float, float] | None:
    """Centre and a size proxy, in viewBox units."""
    g = lambda k, d=0.0: float(el.get(k, d)) if _NUM.fullmatch(
        str(el.get(k, "")).strip() or "x") else d
    try:
        if tag == "rect":
            x, y, w, h = g("x"), g("y"), g("width"), g("height")
            return x + w / 2, y + h / 2, w * h
        if tag == "circle":
            r = g("r")
            return g("cx"), g("cy"), 3.14159 * r * r
        if tag == "ellipse":
            rx, ry = g("rx"), g("ry")
            return g("cx"), g("cy"), 3.14159 * rx * ry
        if tag == "line":
            x1, y1, x2, y2 = g("x1"), g("y1"), g("x2"), g("y2")
            return (x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1) * abs(y2 - y1) + 1e-3
        pts = el.get("points") or el.get("d") or ""
        nums = [float(x) for x in _NUM.findall(pts)]
        if len(nums) < 4:
            return None
        xs, ys = nums[0::2], nums[1::2]
        return (sum(xs) / len(xs), sum(ys) / len(ys),
                (max(xs) - min(xs)) * (max(ys) - min(ys)) + 1e-3)
    except Exception:
        return None


def marks(doc: str) -> list[str] | None:
    """The set of quantized marks the document makes. Deterministic."""
    from lxml import etree
    try:
        parser = etree.XMLParser(recover=True, resolve_entities=False,
                                 no_network=True, huge_tree=False)
        root = etree.fromstring(doc.encode(), parser=parser)
    except Exception:
        return None
    if root is None:
        return None
    # normalize by the declared viewBox so documents drawn at other scales compare
    vb = [float(x) for x in _NUM.findall(root.get("viewBox") or "")] or [0, 0, VIEW, VIEW]
    if len(vb) < 4 or vb[2] <= 0 or vb[3] <= 0:
        vb = [0, 0, VIEW, VIEW]
    out: set[str] = set()
    for el in root.iter():
        tag = etree.QName(el).localname if isinstance(el.tag, str) else ""
        if tag not in _DRAW:
            continue
        e = _extent(el, tag)
        if e is None:
            continue
        cx, cy, area = e
        u = (cx - vb[0]) / vb[2]
        v = (cy - vb[1]) / vb[3]
        if not (-0.5 <= u <= 1.5 and -0.5 <= v <= 1.5):
            continue
        gx = min(GRID - 1, max(0, int(u * GRID)))
        gy = min(GRID - 1, max(0, int(v * GRID)))
        frac = area / (vb[2] * vb[3])
        band = sum(1 for s in SIZE_BANDS if frac > s)
        hue = _hue_band(el.get("fill") or el.get("stroke"))
        out.add(f"{tag}@{gx},{gy}|s{band}|h{hue}")
    return sorted(out) if out else None


def execute() -> None:
    rows = [json.loads(l) for l in (OUT / "corpus.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("safe")]
    with ThreadPoolExecutor(max_workers=8) as ex:
        ms = list(ex.map(lambda r: marks(r["svg"]), rows))
    keep = [{"svg": r["svg"], "marks": m} for r, m in zip(rows, ms) if m]
    with (OUT / "executed.jsonl").open("w") as fh:
        for k in keep:
            fh.write(json.dumps(k) + "\n")
    n = [len(k["marks"]) for k in keep]
    print(f"parsed {len(rows)}, drew marks {len(keep)}; "
          f"median {int(np.median(n)) if n else 0} marks per document")


def mark_distance(ms: list[list[str]]) -> np.ndarray:
    sets = [set(m) for m in ms]
    n = len(sets)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            u = len(a | b)
            D[i, j] = D[j, i] = 1.0 - (len(a & b) / u if u else 1.0)
    return D


def mark_coverage(ms: list[list[str]]):
    sets = [set(m) for m in ms]

    def cov(idx) -> float:
        s: set = set()
        for i in idx:
            s |= sets[i]
        return float(len(s))
    return cov


def _print_purity(pur: dict) -> None:
    print(f"\n  reject purity (pool mean artifact distance "
          f"{pur['pool_mean_artifact_distance']:.3f})")
    print("   radius q      t     rejected   still far   mean art.dist")
    for r in pur["rows"]:
        print(f"     {r['quantile']:.2f}      {r['t']:.3f}  {r['n_rejected']:>9}"
              f"     {100*r['frac_still_far']:5.1f}%      {r['mean_artifact_distance']:.3f}")


def probe(n_bins: int = 10) -> None:
    import probe as P
    from pipeline import embed
    rows = [json.loads(l) for l in (OUT / "executed.jsonl").read_text().splitlines()]
    seen, uniq = set(), []
    for r in rows:
        k = "".join(r["svg"].split())
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq
    docs = [r["svg"] for r in rows]
    ms = [r["marks"] for r in rows]
    print(f"{len(rows)} distinct documents; "
          f"{len({tuple(m) for m in ms})} distinct drawings")

    E = embed(docs)
    TD = P.pairwise_cosine(E)
    AD = mark_distance(ms)
    prof = P.near_field_profile(None, AD, text_dist=TD, n_bins=n_bins)
    P.print_profile(prof, "svg -> marks drawn")
    dd = P.dedup_quality(TD, AD)
    print(f"\n  identical-drawing pairs {dd['n_duplicate_pairs']}; AUC {dd['auc']}")
    cov = mark_coverage(ms)
    budgets = [b for b in (10, 20, 30, 50, 100) if b < len(rows)]
    sel = P.selector_comparison(TD, cov, budgets, n_seeds=20)
    names = (["random", "maxmin"]
             + [a for a in sel["arms"] if a.startswith("filter@")]
             + (["oracle"] if "oracle" in sel["arms"] else []))
    print("\n  marks covered (mean of 20 seeds)")
    print("   k     " + "".join(f"{a:>16}" for a in names))
    for k in budgets:
        print(f"  {k:>3}    " + "".join(f"{sel['arms'][a][k]['mean']:>16.1f}" for a in names))
    pur = P.reject_purity(TD, AD)
    _print_purity(pur)
    rep = P.report(prof, dd, sel, "svg->marks", pur)
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(f"\n  wrote {OUT/'report.json'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if cmd == "generate":
        generate(int(sys.argv[2]) if len(sys.argv) > 2 else 700)
    elif cmd == "execute":
        execute()
    else:
        probe()
