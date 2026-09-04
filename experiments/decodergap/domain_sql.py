"""Domain: generated SQL, decoded by running it against a real database.

Text-to-SQL is one of the largest synthetic-data industries there is, and the
artifact of a generated query is not its text. It is the rows it returns. Two
queries phrased very differently can return identical rows; two queries
differing by one token can return disjoint ones.

The database is fixed and seeded deterministically, the connection is
read-only, and the result set is canonicalized (sorted rows, sorted columns
where the query does not impose an order) so that two queries agree exactly
when they retrieve the same data. Behavioural distance is the Jaccard distance
between result-row sets; coverage is the number of distinct rows a subset of
queries retrieves between them, which is what a synthetic evaluation suite for
a database is actually buying.

    python3 domain_sql.py generate 700
    python3 domain_sql.py execute
    python3 domain_sql.py probe
"""
from __future__ import annotations

import json
import pathlib
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "runs" / "sql"
OUT.mkdir(parents=True, exist_ok=True)
DB = OUT / "shop.db"

sys.path.insert(0, str(HERE))

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY, name TEXT, city TEXT, country TEXT,
    signup_year INTEGER, is_active INTEGER);
CREATE TABLE products (
    id INTEGER PRIMARY KEY, name TEXT, category TEXT,
    price REAL, stock INTEGER);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY, customer_id INTEGER, order_year INTEGER,
    status TEXT, total REAL);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER,
    quantity INTEGER, unit_price REAL);
"""

SCHEMA_DOC = """Tables:
  customers(id, name, city, country, signup_year, is_active)
  products(id, name, category, price, stock)
  orders(id, customer_id, order_year, status, total)
  order_items(id, order_id, product_id, quantity, unit_price)
Categories are one of: kitchen, garden, office, sport, toys.
Order status is one of: paid, pending, refunded, cancelled.
Countries: UK, France, Germany, Spain, Italy. Years run 2019-2024."""

PROMPT = (
    "Here is a SQLite schema.\n\n" + SCHEMA_DOC + "\n\n"
    "Write ONE interesting analytical SELECT query against it. Read-only: no "
    "INSERT, UPDATE, DELETE, PRAGMA, ATTACH or CTE that writes. Return at most "
    "50 rows. Reply with the query only, in a ```sql code block, no commentary."
)


def build_db() -> None:
    if DB.exists():
        DB.unlink()
    rng = np.random.default_rng(20260904)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    cities = ["London", "Paris", "Berlin", "Madrid", "Rome", "Leeds", "Lyon"]
    countries = ["UK", "France", "Germany", "Spain", "Italy"]
    cats = ["kitchen", "garden", "office", "sport", "toys"]
    stat = ["paid", "pending", "refunded", "cancelled"]
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", [
        (i, f"cust{i:03d}", cities[int(rng.integers(len(cities)))],
         countries[int(rng.integers(len(countries)))],
         int(rng.integers(2019, 2025)), int(rng.integers(0, 2)))
        for i in range(1, 121)])
    con.executemany("INSERT INTO products VALUES (?,?,?,?,?)", [
        (i, f"prod{i:03d}", cats[int(rng.integers(len(cats)))],
         float(round(rng.uniform(3, 300), 2)), int(rng.integers(0, 400)))
        for i in range(1, 81)])
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", [
        (i, int(rng.integers(1, 121)), int(rng.integers(2019, 2025)),
         stat[int(rng.integers(len(stat)))], float(round(rng.uniform(5, 2000), 2)))
        for i in range(1, 501)])
    con.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", [
        (i, int(rng.integers(1, 501)), int(rng.integers(1, 81)),
         int(rng.integers(1, 9)), float(round(rng.uniform(3, 300), 2)))
        for i in range(1, 1501)])
    con.commit()
    con.close()
    print(f"built {DB}")


_BAD = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|"
                  r"pragma|vacuum|replace|reindex|begin|commit|rollback)\b", re.I)


def is_safe(q: str) -> tuple[bool, str]:
    s = q.strip().rstrip(";")
    if ";" in s:
        return False, "multiple statements"
    if not re.match(r"^\s*(select|with)\b", s, re.I):
        return False, "not a select"
    if _BAD.search(s):
        return False, "write keyword"
    return True, ""


def extract(text: str) -> str | None:
    """Pull the query out of a fenced block, a bare reply, or surrounding prose."""
    text = text or ""
    for p in text.split("```")[1:]:
        body = p[3:] if p.lower().startswith("sql") else p
        body = body.strip()
        if re.match(r"^\s*(select|with)\b", body, re.I):
            return body
    t = text.strip()
    if re.match(r"^\s*(select|with)\b", t, re.I):
        return t
    m = re.search(r"(?is)\b(with|select)\b.*", t)   # prose, then the query
    return m.group(0).strip() if m else None


def generate(n: int) -> None:
    from pipeline import chat

    def one(i: int) -> dict:
        try:
            t = chat([{"role": "user", "content": PROMPT}],
                     temperature=1.0, max_tokens=1400)
            if not (t or "").strip():   # reasoning models can spend the whole
                t = chat([{"role": "user", "content": PROMPT}],  # budget thinking
                         temperature=1.0, max_tokens=2600)
        except Exception as e:
            return {"i": i, "error": str(e)[:200]}
        q = extract(t or "")
        if not q:
            return {"i": i, "error": "no sql block"}
        ok, why = is_safe(q)
        return {"i": i, "sql": q, "safe": ok, "reject": why}

    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(one, range(n)))
    with (OUT / "corpus.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"generated {len(rows)}, safe {sum(1 for r in rows if r.get('safe'))}")


def run_query(q: str, limit: int = 200) -> list[str] | None:
    """Canonical result: sorted list of row strings. Deterministic by construction."""
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
        con.execute("PRAGMA query_only = ON")
        cur = con.execute(q)
        rows = cur.fetchmany(limit)
        cols = [d[0] for d in cur.description] if cur.description else []
        con.close()
    except Exception:
        return None
    # The artifact is the DATA the query surfaces, recorded as the set of
    # (column, value) cells rather than as whole rows. Whole-row strings are
    # degenerate here: two queries that select different column combinations
    # never share a row even when they read the same records, so row-level
    # Jaccard saturates at 1 for almost every pair and measures the SELECT list
    # rather than the data. Cells do not saturate and answer the question a
    # generated query suite is really being scored on -- which of the database
    # did these queries between them reach.
    out = set()
    for r in rows:
        for c, v in zip(cols or range(len(r)), r):
            if isinstance(v, float):
                s = f"{v:.6g}"
            else:
                s = "NULL" if v is None else str(v)
            out.add(f"{c}={s}")
    return sorted(out) if out else ["<empty>"]


def execute() -> None:
    rows = [json.loads(l) for l in (OUT / "corpus.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("safe")]
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda r: run_query(r["sql"]), rows))
    keep = [{"sql": r["sql"], "rows": v} for r, v in zip(rows, res) if v is not None]
    with (OUT / "executed.jsonl").open("w") as fh:
        for k in keep:
            fh.write(json.dumps(k) + "\n")
    n_empty = sum(1 for k in keep if k["rows"] == ["<empty>"])
    print(f"executed {len(rows)}, ran {len(keep)}, empty result {n_empty}")


def result_distance(res: list[list[str]]) -> np.ndarray:
    """Jaccard distance between result-row sets."""
    sets = [set(r) for r in res]
    n = len(sets)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            u = len(a | b)
            d = 1.0 - (len(a & b) / u if u else 1.0)
            D[i, j] = D[j, i] = d
    return D


def row_coverage(res: list[list[str]]):
    sets = [set(r) - {"<empty>"} for r in res]

    def cov(idx) -> float:
        s: set = set()
        for i in idx:
            s |= sets[i]
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
    seen, uniq = set(), []
    for r in rows:
        k = " ".join(r["sql"].split()).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq
    qs = [r["sql"] for r in rows]
    res = [r["rows"] for r in rows]
    print(f"{len(rows)} distinct queries; "
          f"{len({tuple(r) for r in res})} distinct result sets")

    E = embed(qs)
    TD = P.pairwise_cosine(E)
    AD = result_distance(res)
    prof = P.near_field_profile(None, AD, text_dist=TD, n_bins=n_bins)
    P.print_profile(prof, "sql -> result set")

    dd = P.dedup_quality(TD, AD)
    print(f"\n  identical-result pairs {dd['n_duplicate_pairs']} "
          f"({100*dd['duplicate_pair_rate']:.2f}%); AUC {dd['auc']}")
    if dd["best"]:
        b = dd["best"]
        print(f"  best radius t={b['t']:.3f}  P {b['precision']:.3f} "
              f"R {b['recall']:.3f}  F1 {b['f1']:.3f}")

    cov = row_coverage(res)
    budgets = [b for b in (10, 20, 30, 50, 100) if b < len(rows)]
    sel = P.selector_comparison(TD, cov, budgets, n_seeds=20)
    names = (["random", "maxmin"]
             + [a for a in sel["arms"] if a.startswith("filter@")]
             + (["oracle"] if "oracle" in sel["arms"] else []))
    print("\n  distinct rows retrieved (mean of 20 seeds)")
    print("   k     " + "".join(f"{a:>16}" for a in names))
    for k in budgets:
        print(f"  {k:>3}    " + "".join(f"{sel['arms'][a][k]['mean']:>16.1f}" for a in names))

    pur = P.reject_purity(TD, AD)
    _print_purity(pur)
    rep = P.report(prof, dd, sel, "sql->result set", pur)
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(f"\n  wrote {OUT/'report.json'}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if cmd == "build":
        build_db()
    elif cmd == "generate":
        build_db()
        generate(int(sys.argv[2]) if len(sys.argv) > 2 else 700)
    elif cmd == "execute":
        execute()
    else:
        probe()
