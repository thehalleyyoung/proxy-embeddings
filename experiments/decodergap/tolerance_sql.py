"""The tolerance manipulation on a second decoder, with all four cells.

The cross-terms in the programs tolerance experiment carry the paper's strongest
prescription -- put the tolerance in the ask, because failures are wholesale
rather than near-misses -- and they rest on one decoder. This runs the same
four-cell design on SQL, whose emergent targets are numeric values of computed
aliases (a revenue, an average, a share) and therefore admit a band in exactly
the way a program's return value does.

Only EMERGENT targets are used, by the same schema-fixed rule as elsewhere: a
target column that is not a column of the base schema is an alias for a computed
expression. Constructible targets are excluded because a band around a city name
is meaningless.

    python3 tolerance_sql.py run 3
"""
from __future__ import annotations

import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "runs" / "tolerance_sql2"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent / "code"))

BASE = {"id","name","city","country","signup_year","is_active",
        "category","price","stock","customer_id","order_year","status","total",
        "order_id","product_id","quantity","unit_price"}
TOL = 0.10


def numeric(cell):
    col, val = cell.split("=", 1)
    try:
        return col, float(val)
    except ValueError:
        return col, None


def band(v):
    d = max(abs(v) * TOL, 0.5)
    return v - d, v + d


def prompt_for(req):
    import domain_sql as S
    return S.PROMPT.replace(
        "Reply with the query only",
        f"**Requirement: {req}.** The query must still be a genuine analytical "
        f"query, not a literal SELECT of that value.\n"
        "Reply with the query only")


def _one(args):
    import domain_sql as S
    from pipeline import chat
    arm, t, seed = args
    col, v = numeric(t["cell"])
    lo, hi = band(v)
    rec = {"arm": arm, "band": t["band"], "rarity": t["rarity"],
           "cell": t["cell"], "seed": seed}
    if arm == "exact":
        req = (f"the result must contain at least one row in which the column "
               f"`{col}` has the value `{v:g}`")
    else:
        req = (f"the result must contain at least one row in which the column "
               f"`{col}` has a value between {lo:.4g} and {hi:.4g}")
    try:
        txt = chat([{"role": "user", "content": prompt_for(req)}],
                   temperature=1.0, max_tokens=2600)
        if not (txt or "").strip():
            txt = chat([{"role": "user", "content": prompt_for(req)}],
                       temperature=1.0, max_tokens=4000)
        q = S.extract(txt or "")
        if q and S.is_safe(q)[0]:
            rows = S.run_query(q)
            if rows is not None:
                vals = []
                for c in rows:
                    cc, vv = numeric(c)
                    if cc == col and vv is not None:
                        vals.append(vv)
                rec["n_vals"] = len(vals)
                rec["hit_exact"] = any(abs(x - v) < 1e-6 for x in vals)
                rec["hit_tol"] = any(lo <= x <= hi for x in vals)
    except Exception as e:
        rec["error"] = str(e)[:160]
    return rec


def run(n_seeds=3):
    tg = json.loads((HERE / "runs" / "lever_sql" / "targets.json").read_text())
    tg = [t for t in tg
          if t["cell"].split("=")[0] not in BASE and numeric(t["cell"])[1] is not None]
    print(f"{len(tg)} emergent numeric targets")
    jobs = [(a, t, s) for a in ("exact", "tolerant") for t in tg for s in range(n_seeds)]
    print(f"{len(jobs)} jobs")
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(_one, jobs))
    with (OUT / "results.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    report(rows)


def report(rows=None):
    if rows is None:
        rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text().splitlines()]
    def ci(h):
        h = np.array(h, dtype=float); rng = np.random.default_rng(0)
        bs = [rng.choice(h, len(h)).mean() for _ in range(4000)]
        return h.mean(), np.percentile(bs, [2.5, 97.5]), len(h)
    print("\n  SQL, emergent numeric targets: how asked x how scored")
    for arm, key, label in (("exact","hit_exact","asked exactly, scored exactly"),
                            ("tolerant","hit_tol","asked as a band, scored on the band"),
                            ("tolerant","hit_exact","asked as a band, scored exactly"),
                            ("exact","hit_tol","asked exactly, scored on the band")):
        h=[bool(r.get(key)) for r in rows if r["arm"]==arm]
        if not h: continue
        m,(lo,hi),n=ci(h)
        print(f"    {label:<40} {100*m:5.1f}%  [{100*lo:4.1f},{100*hi:5.1f}]  n={n}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        report()
