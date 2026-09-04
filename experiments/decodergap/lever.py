"""The lever, on every decoder: can a target be put INTO the control modality?

`steer.py` established on generated Python programs that a target stated in the
text is hit far more often than a decoy target stated in identical form. That
result rests on one decoder, and one decoder is not a finding. This module runs
the same four arms on three, through a small per-domain adapter, and adds the
band `steer.py` could not supply: targets the generator has NEVER produced.

An adapter supplies four things and nothing else:
  `corpus()`      the natural corpus, as (texts, artifacts)
  `cells(a)`      the set of target-cells an artifact satisfies
  `render(c)`     the cell written into a prompt requirement
  `base()`        the unconditioned prompt

Rarity is the share of the natural corpus already satisfying a cell, so it is
measured in the generator's own artifact distribution rather than assumed.

    python3 lever.py targets sql
    python3 lever.py run sql 3
    python3 lever.py report sql
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "code"))

SEED = 20260904
PER_BAND = 12
BANDS = [(0.001, 0.02), (0.02, 0.10), (0.10, 0.30), (0.30, 1.01)]


# --------------------------------------------------------------- adapters

class Sql:
    name = "sql"

    def corpus(self):
        import domain_sql as S
        rows = [json.loads(l) for l in
                (S.OUT / "executed.jsonl").read_text().splitlines()]
        seen, uniq = set(), []
        for r in rows:
            k = " ".join(r["sql"].split()).lower()
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        return [r["sql"] for r in uniq], [r["rows"] for r in uniq]

    def cells(self, a):
        return {c for c in a if c != "<empty>"}

    def render(self, c):
        return (f"the result must contain at least one row in which the column "
                f"`{c.split('=')[0]}` has the value `{c.split('=', 1)[1]}`")

    def base(self):
        import domain_sql as S
        return S.PROMPT

    def prompt(self, req):
        import domain_sql as S
        return (S.PROMPT.replace(
            "Reply with the query only",
            f"**Requirement: {req}.** The query must still be a genuine "
            f"analytical query, not a literal SELECT of that value.\n"
            "Reply with the query only"))

    def decode(self, text):
        import domain_sql as S
        if not S.is_safe(text)[0]:
            return None
        return S.run_query(text)

    def extract(self, t):
        import domain_sql as S
        return S.extract(t)

    def unobserved(self, texts, arts, rng, k):
        """Cells no query returned: a real column with a value from the database."""
        import domain_sql as S
        seen = set()
        for a in arts:
            seen |= self.cells(a)
        cols = collections.Counter(c.split("=")[0] for c in seen)
        out = []
        con = __import__("sqlite3").connect(f"file:{S.DB}?mode=ro", uri=True)
        pool = []
        for col, tbl in (("city", "customers"), ("country", "customers"),
                         ("category", "products"), ("status", "orders"),
                         ("name", "products"), ("name", "customers")):
            try:
                for (v,) in con.execute(f"SELECT DISTINCT {col} FROM {tbl}"):
                    pool.append(f"{col}={v}")
            except Exception:
                pass
        con.close()
        rng.shuffle(pool)
        for c in pool:
            if c not in seen and cols.get(c.split("=")[0], 0) >= 0:
                out.append(c)
            if len(out) >= k:
                break
        return out


class Regex:
    name = "regex"

    def corpus(self):
        import domain_regex as R
        rows = [json.loads(l) for l in
                (R.OUT / "executed.jsonl").read_text().splitlines()]
        seen, uniq = set(), []
        for r in rows:
            if r["pattern"] not in seen:
                seen.add(r["pattern"])
                uniq.append(r)
        return [r["pattern"] for r in uniq], [r["hits"] for r in uniq]

    def cells(self, a):
        return set(a)

    def render(self, c):
        import domain_regex as R
        return f"the pattern must match the string {R.PROBES[c]!r}"

    def base(self):
        import domain_regex as R
        return R.PROMPT

    def prompt(self, req):
        import domain_regex as R
        return R.PROMPT.replace(
            "Reply with the pattern only",
            f"**Requirement: {req}.** It must still be a general, meaningful "
            f"pattern, not a literal spelling of that one string.\n"
            "Reply with the pattern only")

    def decode(self, text):
        import domain_regex as R
        return R.run_one(text)

    def extract(self, t):
        import domain_regex as R
        p = R.extract(t)
        if p and p.startswith(("r'", 'r"')):
            p = p[2:-1] if p[-1] in "'\"" else p[2:]
        elif p and p[0] in "'\"" and p[-1] == p[0]:
            p = p[1:-1]
        return p

    def unobserved(self, texts, arts, rng, k):
        import domain_regex as R
        seen = set()
        for a in arts:
            seen |= set(a)
        pool = [i for i in range(len(R.PROBES)) if i not in seen and R.PROBES[i]]
        rng.shuffle(pool)
        return pool[:k]


class Svg:
    """SVG marks. Every target is constructible AND observable: the required
    element, cell, size band and hue band are all stated, and the generator can
    write exactly that element. This is the positive control for the rule."""
    name = "svg"

    GRID_NAMES = ["far left", "left", "centre-left", "centre-right", "right",
                  "far right"]
    ROW_NAMES = ["top", "upper", "upper-middle", "lower-middle", "lower",
                 "bottom"]
    SIZE_NAMES = ["tiny", "small", "medium", "large", "very large"]
    HUE_NAMES = ["red", "orange", "yellow", "green", "cyan", "blue", "purple",
                 "magenta", "grey or black or white"]

    def corpus(self):
        import domain_svg as V
        rows = [json.loads(l) for l in
                (V.OUT / "executed.jsonl").read_text().splitlines()]
        seen, uniq = set(), []
        for r in rows:
            k = "".join(r["svg"].split())
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        return [r["svg"] for r in uniq], [r["marks"] for r in uniq]

    def cells(self, a):
        return set(a)

    def render(self, c):
        tag, rest = c.split("@", 1)
        pos, sz, hue = rest.split("|")
        gx, gy = (int(v) for v in pos.split(","))
        return (f"the drawing must contain a `<{tag}>` whose centre falls in the "
                f"{self.ROW_NAMES[gy]} {self.GRID_NAMES[gx]} region of the "
                f"viewBox (grid cell column {gx+1} of 6, row {gy+1} of 6), which "
                f"is {self.SIZE_NAMES[int(sz[1:])]} in area, and whose fill is "
                f"{self.HUE_NAMES[int(hue[1:])]}")

    def base(self):
        import domain_svg as V
        return V.PROMPT

    def prompt(self, req):
        import domain_svg as V
        return V.PROMPT.replace(
            "Reply with the SVG only",
            f"**Requirement: {req}.** The drawing must still be a coherent "
            f"picture, not a single element on its own.\n"
            "Reply with the SVG only")

    def decode(self, text):
        import domain_svg as V
        if not V.is_safe(text)[0]:
            return None
        return V.marks(text)

    def extract(self, t):
        import domain_svg as V
        return V.extract(t)

    def unobserved(self, texts, arts, rng, k):
        import domain_svg as V
        seen = set()
        for a in arts:
            seen |= set(a)
        pool = [f"{t}@{x},{y}|s{s}|h{h}"
                for t in ("rect", "circle", "ellipse", "polygon", "path", "line")
                for x in range(6) for y in range(6)
                for s in range(len(V.SIZE_BANDS) + 1) for h in range(V.HUES + 1)]
        pool = [c for c in pool if c not in seen]
        rng.shuffle(pool)
        return pool[:k]


ADAPTERS = {"sql": Sql(), "regex": Regex(), "svg": Svg()}


def out_dir(name: str) -> pathlib.Path:
    d = HERE / "runs" / f"lever_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- targets

def build_targets(name: str) -> None:
    A = ADAPTERS[name]
    texts, arts = A.corpus()
    n = len(texts)
    counts: collections.Counter = collections.Counter()
    for a in arts:
        for c in A.cells(a):
            counts[c] += 1
    print(f"[{name}] {n} items, {len(counts)} distinct cells")

    rng = random.Random(SEED)
    targets = []
    for c in A.unobserved(texts, arts, rng, PER_BAND):
        targets.append({"cell": c, "rarity": 0.0, "band": "[0,0.001) unobserved",
                        "req": A.render(c)})
    print(f"  unobserved: {len(targets)}")
    for lo, hi in BANDS:
        pool = [c for c, k in counts.items() if lo <= k / n < hi]
        rng.shuffle(pool)
        for c in pool[:PER_BAND]:
            targets.append({"cell": c, "rarity": counts[c] / n,
                            "band": f"[{lo:g},{hi:g})", "req": A.render(c)})
        print(f"  [{lo:g},{hi:g}): {len(pool)} available, {min(len(pool), PER_BAND)} taken")
    (out_dir(name) / "targets.json").write_text(json.dumps(targets, indent=2))
    print(f"  wrote {len(targets)} targets")


# -------------------------------------------------------------------- run

def _one(args) -> dict:
    from pipeline import chat
    name, arm, t, seed, decoy = args
    A = ADAPTERS[name]
    rec = {"arm": arm, "band": t["band"], "rarity": t["rarity"],
           "cell": str(t["cell"]), "seed": seed}
    try:
        if arm == "blind":
            prompt = A.base()
        elif arm == "decoy":
            prompt = A.prompt(decoy["req"])
        else:
            prompt = A.prompt(t["req"])
        txt = chat([{"role": "user", "content": prompt}],
                   temperature=1.0, max_tokens=1400)
        if not (txt or "").strip():
            txt = chat([{"role": "user", "content": prompt}],
                       temperature=1.0, max_tokens=2600)
        item = A.extract(txt or "")
        if item:
            a = A.decode(item)
            if a is not None:
                cs = A.cells(a)
                rec["hit"] = t["cell"] in cs
                rec["n_cells"] = len(cs)
                # the produced artifact is kept so a miss can be diagnosed:
                # a target that is a CONJUNCTION of quantized properties can be
                # missed by a hair on one of them, and exact scoring cannot tell
                # that from missing it entirely.
                if name == "svg":
                    rec["produced"] = sorted(cs)[:400]
    except Exception as e:
        rec["error"] = str(e)[:160]
    return rec


def run(name: str, n_seeds: int = 3) -> None:
    d = out_dir(name)
    targets = json.loads((d / "targets.json").read_text())
    rng = random.Random(SEED)
    jobs = []
    for arm in ("blind", "decoy", "instruct"):
        for t in targets:
            pool = [u for u in targets if u["cell"] != t["cell"]]
            for s in range(n_seeds):
                jobs.append((name, arm, t, s, rng.choice(pool)))
    print(f"[{name}] {len(jobs)} jobs")
    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(_one, jobs))
    with (d / "results.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows)}")


def report(name: str) -> None:
    d = out_dir(name)
    rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines()]
    arms = ["blind", "decoy", "instruct"]

    def rate(sel):
        h = [bool(r.get("hit")) for r in sel]
        if not h:
            return float("nan"), 0, (float("nan"), float("nan"))
        rng = np.random.default_rng(0)
        bs = [np.mean(rng.choice(h, len(h))) for _ in range(2000)]
        return float(np.mean(h)), len(h), tuple(np.percentile(bs, [2.5, 97.5]))

    bands = sorted({r["band"] for r in rows},
                   key=lambda b: float(b.split(",")[0].lstrip("[")))
    print(f"\n  [{name}] compliance, checked by decoding")
    print("  rarity band              n" + "".join(f"{a:>24}" for a in arms))
    for b in bands:
        cells, n0 = [], 0
        for a in arms:
            p, n, ci = rate([r for r in rows if r["band"] == b and r["arm"] == a])
            n0 = n0 or n
            cells.append(f"{100*p:>10.1f}% [{100*ci[0]:.0f},{100*ci[1]:.0f}]")
        print(f"  {b:<22}{n0:>3}" + "".join(f"{c:>24}" for c in cells))
    print("\n  overall")
    for a in arms:
        p, n, ci = rate([r for r in rows if r["arm"] == a])
        print(f"    {a:<10} {100*p:>6.1f}%  [{100*ci[0]:.1f}, {100*ci[1]:.1f}]  n={n}")
    json.dump({"n": len(rows)}, open(d / "summary.json", "w"))


if __name__ == "__main__":
    cmd, name = sys.argv[1], sys.argv[2]
    if cmd == "targets":
        build_targets(name)
    elif cmd == "run":
        run(name, int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        report(name)
