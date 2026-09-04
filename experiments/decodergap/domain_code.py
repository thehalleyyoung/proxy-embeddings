"""Domain: generated Python programs, decoded by running them.

The artifact of a synthetic code corpus is not its source. It is what the
programs do. This module builds a corpus of single-function programs against a
fixed signature, executes each on a fixed battery of inputs, and takes the
vector of results as the artifact.

Two programs are behaviourally identical when they agree on the whole battery,
whatever their source looks like; behavioural distance is the fraction of the
battery on which they disagree. That is a genuine metric on the artifact, it is
deterministic, and it is computed without reference to any embedding — so it
can adjudicate an embedding rather than agree with it.

The signature is fixed (`def f(xs)` over a list of ints returning an int) so
one battery applies to every program in the corpus. Generation is naive
repeated prompting, the honest floor for a synthetic-data pipeline and the
setting in which near-duplicates actually arise.

    python3 domain_code.py generate 600
    python3 domain_code.py execute
    python3 domain_code.py probe
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "runs" / "code"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))

MODEL = os.environ.get("DG_MODEL", "openai/gpt-5.6-luna")
PROMPT = (
    "Write one Python function with exactly this signature:\n\n"
    "    def f(xs):\n\n"
    "`xs` is a list of integers. Return a single integer. The function must be "
    "pure: no imports, no printing, no randomness, no I/O, no global state. "
    "Make it do something specific and interesting with the list. "
    "Reply with the function only, in a ```python code block, no commentary."
)


# ------------------------------------------------------------ the battery

def battery() -> list[list[int]]:
    """A fixed, deterministic set of inputs every program is scored on."""
    rng = np.random.default_rng(20260904)
    cases: list[list[int]] = [
        [], [0], [1], [-1], [7], [0, 0], [1, 1], [1, 2], [2, 1], [-1, 1],
        [1, 2, 3], [3, 2, 1], [1, 1, 1], [0, 0, 0], [-3, -2, -1],
        [5, 5, 5, 5], [1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [2, 4, 6, 8],
        [1, 3, 5, 7], [10, -10, 10, -10], [100, 1, 100, 1],
        [0, 1, 0, 1, 0], [-5, 0, 5], [1000000, 1], [1, 1000000],
        [2, 3, 5, 7, 11, 13], [4, 6, 8, 9, 10], [1, 0, -1, 0, 1],
        [12, 15, 18, 21], [7, 7, 8, 8, 9], [-2, 4, -8, 16],
    ]
    for n in (2, 3, 5, 8, 13, 21):
        for _ in range(14):
            cases.append([int(x) for x in rng.integers(-50, 51, size=n)])
    for n in (4, 6, 10):
        for _ in range(8):
            cases.append([int(x) for x in rng.integers(0, 10, size=n)])
    return cases


BATTERY = battery()

# ---------------------------------------------------------- static safety

_ALLOWED_CALLS = {
    "abs", "all", "any", "bin", "bool", "divmod", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "pow", "range", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip", "dict", "frozenset",
    "f", "ord", "chr",
}
_ALLOWED_METHODS = {
    "append", "count", "extend", "index", "insert", "items", "keys", "pop",
    "remove", "reverse", "sort", "values", "get", "add", "update", "join",
    "isdigit", "bit_length", "copy", "difference", "intersection", "union",
    "setdefault", "discard", "clear", "startswith", "endswith", "lower", "upper",
    "split", "strip", "replace", "find", "format",
}


def is_safe(src: str) -> tuple[bool, str]:
    """Reject anything that could reach outside the arithmetic sandbox.

    Calls are allowed to the numeric builtins whitelist and to names the
    program itself defines (nested helpers and recursion are common and
    harmless); everything else -- imports, dunder access, unknown methods --
    is refused, so `open`, `eval` and `__import__` have no route in.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return False, f"syntax: {e}"
    local = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            local |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        if isinstance(n, (ast.For, ast.comprehension)):
            tgt = n.target
            local |= {x.id for x in ast.walk(tgt) if isinstance(x, ast.Name)}
    allowed_calls = _ALLOWED_CALLS | local
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "import"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return False, "global"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr not in _ALLOWED_METHODS:
                return False, f"attribute {node.attr}"
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            return False, f"dunder {node.id}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in allowed_calls:
                return False, f"call {node.func.id}"
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not any(n.name == "f" for n in fns):
        return False, "no def f"
    top = {type(n) for n in tree.body}
    if not top <= {ast.FunctionDef, ast.Expr}:
        return False, "top-level statement"
    return True, ""


# ------------------------------------------------------------- generation

def extract(text: str) -> str | None:
    if "```" not in text:
        return text.strip() if "def f(" in text else None
    parts = text.split("```")
    for p in parts[1:]:
        body = p[len("python"):] if p.lower().startswith("python") else p
        if "def f(" in body:
            return body.strip()
    return None


def generate(n: int) -> None:
    from pipeline import chat  # noqa: E402  (repo client, OpenRouter + retries)

    def one(i: int) -> dict | None:
        try:
            t = chat([{"role": "user", "content": PROMPT}],
                     temperature=1.0, max_tokens=420)
        except Exception as e:
            return {"i": i, "error": str(e)[:200]}
        src = extract(t or "")
        if not src:
            return {"i": i, "error": "no code block"}
        ok, why = is_safe(src)
        return {"i": i, "src": src, "safe": ok, "reject": why}

    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(one, range(n)))
    path = OUT / "corpus.jsonl"
    with path.open("w") as fh:
        for r in rows:
            if r:
                fh.write(json.dumps(r) + "\n")
    good = [r for r in rows if r and r.get("safe")]
    print(f"generated {len(rows)}, safe {len(good)} -> {path}")


# -------------------------------------------------------------- execution

RUNNER = r'''
import json, sys, resource
try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
except Exception:
    pass
sys.setrecursionlimit(3000)
src, cases = json.loads(sys.stdin.read())
ns = {"__builtins__": __builtins__}
out = []
try:
    exec(src, ns)
    f = ns["f"]
except Exception as e:
    print(json.dumps({"error": "compile: " + type(e).__name__}))
    sys.exit(0)
for c in cases:
    try:
        v = f(list(c))
        if isinstance(v, bool):
            out.append("B" + str(v))
        elif isinstance(v, int):
            out.append("I" + str(v))
        elif isinstance(v, float):
            out.append("F" + repr(round(v, 9)))
        else:
            out.append("X" + type(v).__name__)
    except RecursionError:
        out.append("E:RecursionError")
    except Exception as e:
        out.append("E:" + type(e).__name__)
print(json.dumps({"fp": out}))
'''


def run_one(src: str, timeout: float = 12.0) -> list[str] | None:
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER],
                           input=json.dumps([src, BATTERY]),
                           capture_output=True, text=True, timeout=timeout)
        d = json.loads(p.stdout.strip().splitlines()[-1])
        return d.get("fp")
    except Exception:
        return None


def execute() -> None:
    rows = [json.loads(l) for l in (OUT / "corpus.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("safe")]
    with ThreadPoolExecutor(max_workers=8) as ex:
        fps = list(ex.map(lambda r: run_one(r["src"]), rows))
    keep = [{"src": r["src"], "fp": fp} for r, fp in zip(rows, fps) if fp]
    # a program that raises on every input carries no behaviour to compare
    keep = [k for k in keep if len({v for v in k["fp"]}) > 1
            or not k["fp"][0].startswith("E:")]
    path = OUT / "executed.jsonl"
    with path.open("w") as fh:
        for k in keep:
            fh.write(json.dumps(k) + "\n")
    print(f"executed {len(rows)}, usable {len(keep)} -> {path}")


# ------------------------------------------------------------ measurement

def behavioural_distance(fps: list[list[str]]) -> np.ndarray:
    """Fraction of the battery on which two programs disagree."""
    A = np.array(fps, dtype=object)
    n = len(fps)
    D = np.zeros((n, n))
    for i in range(n):
        eq = (A[i][None, :] == A)
        D[i] = 1.0 - eq.mean(axis=1)
    np.fill_diagonal(D, 0.0)
    return D


def behaviour_cells(fps: list[list[str]]):
    """Coverage: distinct (battery index, result) pairs a subset reaches."""
    cells = [set((j, v) for j, v in enumerate(fp)) for fp in fps]

    def cov(idx) -> float:
        s: set = set()
        for i in idx:
            s |= cells[i]
        return float(len(s))
    return cov



def _print_purity(pur: dict) -> None:
    """What a near-duplicate filter throws away, pair by pair."""
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
    # exact source duplicates are a different phenomenon; keep one of each
    seen, uniq = set(), []
    for r in rows:
        k = r["src"].strip()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq
    srcs = [r["src"] for r in rows]
    fps = [r["fp"] for r in rows]
    print(f"{len(rows)} distinct programs; "
          f"{len({tuple(f) for f in fps})} distinct behaviours")

    E = embed(srcs)
    TD = P.pairwise_cosine(E)
    AD = behavioural_distance(fps)
    prof = P.near_field_profile(None, AD, text_dist=TD, n_bins=n_bins)
    P.print_profile(prof, "code -> execution")

    dd = P.dedup_quality(TD, AD)
    print(f"\n  duplicate pairs {dd['n_duplicate_pairs']} "
          f"({100*dd['duplicate_pair_rate']:.2f}% of pairs); "
          f"detection AUC {dd['auc']:.3f}")
    if dd["best"]:
        b = dd["best"]
        print(f"  best radius t={b['t']:.3f}  precision {b['precision']:.3f} "
              f"recall {b['recall']:.3f}  F1 {b['f1']:.3f}")

    cov = behaviour_cells(fps)
    budgets = [b for b in (10, 20, 30, 50, 100) if b < len(rows)]
    sel = P.selector_comparison(TD, cov, budgets, n_seeds=20)
    print("\n  behaviour cells covered (mean of 20 seeds)")
    names = (["random", "maxmin"]
             + [a for a in sel["arms"] if a.startswith("filter@")]
             + (["oracle"] if "oracle" in sel["arms"] else []))
    print("   k     " + "".join(f"{a:>16}" for a in names))
    for k in budgets:
        print(f"  {k:>3}    " + "".join(f"{sel['arms'][a][k]['mean']:>16.1f}" for a in names))

    pur = P.reject_purity(TD, AD)
    _print_purity(pur)
    rep = P.report(prof, dd, sel, "code->execution", pur)
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(f"\n  wrote {OUT/'report.json'}")
    print("  verdicts:", json.dumps(rep["verdicts"], indent=2, default=float)[:1200])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if cmd == "generate":
        generate(int(sys.argv[2]) if len(sys.argv) > 2 else 600)
    elif cmd == "execute":
        execute()
    else:
        probe()
