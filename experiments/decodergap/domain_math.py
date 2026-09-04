"""Domain: generated arithmetic expressions, decoded to their values on a grid.

This decoder exists to sit at the smooth end of the range. Editing a
coefficient moves the artifact proportionally, where flipping an operator in a
program moves a branch discontinuously and editing one character of a regular
expression can change the language it accepts entirely. If near-field agreement
is a signature of shared continuity rather than of proxy quality, near-field
strength should be largest here and smallest for the discontinuous decoders,
and that prediction is what this domain is for.

The artifact is the vector of values the expression takes on a fixed grid of
inputs. Artifact distance is one minus the correlation of two such vectors after
robust standardization, so it measures whether two expressions describe the same
shape rather than whether they happen to share a scale.

    python3 domain_math.py generate 700
    python3 domain_math.py execute
    python3 domain_math.py probe
"""
from __future__ import annotations

import ast
import json
import math
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "runs" / "math"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))

GRID = np.linspace(-3.0, 3.0, 121)

PROMPT = (
    "Write ONE mathematical expression in the single variable x, as a Python "
    "expression. Use only + - * / ** ( ) numbers and x, and optionally the names "
    "sin, cos, tan, exp, log, sqrt, abs. No assignment, no lambda, no other "
    "names. Make it a specific, interesting function — not just a simple "
    "polynomial. Reply with the expression on one line in a ```python code "
    "block, nothing else."
)

_FUNCS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp,
          "log": math.log, "sqrt": math.sqrt, "abs": abs}


def is_safe(expr: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return False, f"syntax: {e}"
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            if n.id != "x" and n.id not in _FUNCS:
                return False, f"name {n.id}"
        elif isinstance(n, ast.Call):
            if not (isinstance(n.func, ast.Name) and n.func.id in _FUNCS):
                return False, "call"
        elif isinstance(n, (ast.Attribute, ast.Subscript, ast.Lambda,
                            ast.comprehension, ast.IfExp)):
            return False, type(n).__name__
        elif isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
            return False, "non-numeric constant"
    return True, ""


def extract(text: str) -> str | None:
    text = text or ""
    for p in text.split("```")[1:]:
        body = p[6:] if p.lower().startswith("python") else p
        body = body.strip()
        if body:
            return body.splitlines()[0].strip()
    t = text.strip()
    return t.splitlines()[0].strip() if t else None


def generate(n: int) -> None:
    from pipeline import chat

    def one(i: int) -> dict:
        try:
            t = chat([{"role": "user", "content": PROMPT}],
                     temperature=1.0, max_tokens=1200)
            if not (t or "").strip():
                t = chat([{"role": "user", "content": PROMPT}],
                         temperature=1.0, max_tokens=2600)
        except Exception as e:
            return {"i": i, "error": str(e)[:200]}
        e = extract(t)
        if not e:
            return {"i": i, "error": "no expression"}
        ok, why = is_safe(e)
        return {"i": i, "expr": e, "safe": ok, "reject": why}

    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(one, range(n)))
    with (OUT / "corpus.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"generated {len(rows)}, safe {sum(1 for r in rows if r.get('safe'))}")


def evaluate(expr: str) -> list[float] | None:
    """Values on the fixed grid. Non-finite points become NaN, not failures."""
    ok, _ = is_safe(expr)
    if not ok:
        return None
    try:
        code = compile(ast.parse(expr, mode="eval"), "<expr>", "eval")
    except Exception:
        return None
    env = dict(_FUNCS)
    out = []
    for x in GRID:
        env["x"] = float(x)
        try:
            v = eval(code, {"__builtins__": {}}, env)
            v = float(v)
            out.append(v if math.isfinite(v) else math.nan)
        except Exception:
            out.append(math.nan)
    if sum(1 for v in out if math.isfinite(v)) < 0.5 * len(GRID):
        return None
    return out


def execute() -> None:
    rows = [json.loads(l) for l in (OUT / "corpus.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("safe")]
    with ThreadPoolExecutor(max_workers=8) as ex:
        vals = list(ex.map(lambda r: evaluate(r["expr"]), rows))
    keep = [{"expr": r["expr"], "y": v} for r, v in zip(rows, vals) if v is not None]
    # a constant function has no shape to compare
    keep = [k for k in keep
            if np.nanstd(np.array(k["y"], dtype=float)) > 1e-9]
    with (OUT / "executed.jsonl").open("w") as fh:
        for k in keep:
            fh.write(json.dumps(k) + "\n")
    print(f"evaluated {len(rows)}, usable {len(keep)}")


def _z(Y: np.ndarray) -> np.ndarray:
    """Robust standardization: rank-transform, so a scale change is not a shape change."""
    from scipy.stats import rankdata
    Z = np.empty_like(Y)
    for i, row in enumerate(Y):
        m = np.isfinite(row)
        r = np.full(row.shape, np.nan)
        if m.sum() > 1:
            r[m] = rankdata(row[m]) / m.sum()
        Z[i] = r
    return Z


def shape_distance(ys: list[list[float]]) -> np.ndarray:
    """One minus the rank correlation of two value vectors on the shared grid."""
    Y = np.array(ys, dtype=float)
    Z = _z(Y)
    n = len(ys)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = np.isfinite(Z[i]) & np.isfinite(Z[j])
            if m.sum() < 8:
                d = 1.0
            else:
                a, b = Z[i][m], Z[j][m]
                sa, sb = a.std(), b.std()
                c = 0.0 if sa < 1e-12 or sb < 1e-12 else float(
                    ((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
                # (1 - c)/2, not 1 - |c|: a function and its negation are
                # opposite shapes, not the same one, and |c| would rate cos(x)
                # identical to x**2 on a symmetric grid.
                d = (1.0 - c) / 2.0
            D[i, j] = D[j, i] = d
    return D


def shape_coverage(ys: list[list[float]], n_cells: int = 24):
    """Coverage: distinct (grid band, value band) cells a subset of curves visits."""
    Y = np.array(ys, dtype=float)
    Z = _z(Y)
    cells = []
    for row in Z:
        s = set()
        for gi, v in enumerate(row):
            if np.isfinite(v):
                s.add((gi * n_cells // len(row), int(v * n_cells)))
        cells.append(s)

    def cov(idx) -> float:
        s: set = set()
        for i in idx:
            s |= cells[i]
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
        k = "".join(r["expr"].split())
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq
    exprs = [r["expr"] for r in rows]
    ys = [r["y"] for r in rows]
    print(f"{len(rows)} distinct expressions")

    E = embed(exprs)
    TD = P.pairwise_cosine(E)
    AD = shape_distance(ys)
    prof = P.near_field_profile(None, AD, text_dist=TD, n_bins=n_bins)
    P.print_profile(prof, "math expression -> values on a grid")

    dd = P.dedup_quality(TD, AD, identical_tol=1e-6)
    print(f"\n  same-shape pairs {dd['n_duplicate_pairs']}; AUC {dd['auc']}")
    cov = shape_coverage(ys)
    budgets = [b for b in (10, 20, 30, 50, 100) if b < len(rows)]
    sel = P.selector_comparison(TD, cov, budgets, n_seeds=20)
    names = (["random", "maxmin"]
             + [a for a in sel["arms"] if a.startswith("filter@")]
             + (["oracle"] if "oracle" in sel["arms"] else []))
    print("\n  shape cells covered (mean of 20 seeds)")
    print("   k     " + "".join(f"{a:>16}" for a in names))
    for k in budgets:
        print(f"  {k:>3}    " + "".join(f"{sel['arms'][a][k]['mean']:>16.1f}" for a in names))

    pur = P.reject_purity(TD, AD)
    _print_purity(pur)
    rep = P.report(prof, dd, sel, "math->values", pur)
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
