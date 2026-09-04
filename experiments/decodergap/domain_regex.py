"""Domain: generated regular expressions, decoded into the language they accept.

A regex is the sharpest case in the study because the map from text to artifact
is visibly discontinuous in both directions. `a+` and `a*` differ by one
character and accept different languages; `[0-9]+` and `\\d+` share almost no
characters and accept the same one. Any distance computed on the source is
being asked to stand in for a distance between sets of strings, and the two
have no reason to agree.

The artifact is the subset of a fixed probe corpus that the expression matches.
Behavioural distance is the Jaccard distance between those subsets; coverage is
the number of distinct probe strings a set of expressions matches between them,
which is what a generated validation or extraction suite is buying.

Matching runs in a subprocess with a timeout, because a generated expression can
backtrack catastrophically and that is a property of the corpus, not a bug.

    python3 domain_regex.py generate 700
    python3 domain_regex.py execute
    python3 domain_regex.py probe
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "runs" / "regex"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))


def probes() -> list[str]:
    """A fixed corpus of strings every expression is scored against."""
    rng = np.random.default_rng(20260904)
    base = [
        "", " ", "a", "A", "0", "9", "abc", "ABC", "Abc", "a1", "1a", "a_b",
        "hello world", "hello_world", "helloWorld", "HELLO", "  padded  ",
        "3.14", "-3.14", "+42", "0042", "1e10", "0x1f", "1,000", "1_000",
        "2024-01-31", "31/01/2024", "01-31-2024", "2024-13-45", "12:34",
        "12:34:56", "user@example.com", "user.name+tag@sub.example.co.uk",
        "not an email@", "@nouser.com", "http://example.com",
        "https://example.com/path?q=1#frag", "ftp://files.example.org",
        "www.example.com", "192.168.0.1", "999.999.999.999", "::1",
        "fe80::1ff:fe23:4567:890a", "#FFFFFF", "#fff", "#ggg",
        "+44 20 7946 0958", "(555) 123-4567", "555-1234",
        "camelCaseName", "PascalCaseName", "snake_case_name", "kebab-case-name",
        "SCREAMING_SNAKE", "_leading", "trailing_", "__dunder__",
        "def f(x): return x", "SELECT * FROM t;", "<tag>text</tag>",
        "<br/>", "{\"k\": 1}", "[1, 2, 3]", "key=value", "a=1&b=2",
        "/usr/local/bin", "C:\\Windows\\System32", "file.txt", "archive.tar.gz",
        ".hidden", "no-extension", "UPPER.TXT", "a.b.c.d",
        "The quick brown fox.", "Sentence one. Sentence two!", "Really?",
        "tab\there", "new\nline", "trailing space ", "  ", "\t",
        "aaa", "aab", "aba", "baa", "abab", "aabb", "abcabc", "xyz",
        "aaaaaaaaaaaaaaaaaaaa", "ababababababababab", "0000000000",
        "£100", "€50", "$1,234.56", "50%", "3/4", "n/a", "N/A", "NULL", "null",
        "true", "false", "True", "False", "yes", "no", "1", "0",
        "café", "naïve", "日本語", "emoji 🙂", "Ω", "ß",
        "v1.2.3", "v1.2.3-beta.1", "1.0", "2.0.0-rc1",
        "ISBN 978-3-16-148410-0", "AB-1234-CD", "XYZ123", "123XYZ",
    ]
    alph = "abc01 _-."
    for n in (3, 6, 10):
        for _ in range(30):
            base.append("".join(alph[int(i)] for i in rng.integers(0, len(alph), n)))
    return base


PROBES = probes()

PROMPT = (
    "Write ONE Python regular expression that matches a specific, useful "
    "pattern — an identifier, a number format, a date, an address, a code "
    "construct, whatever you think is interesting. Do not use lookbehind. "
    "Reply with the pattern only, on one line, in a ```regex code block, with "
    "no delimiters, no flags, no quotes and no commentary."
)


def extract(text: str) -> str | None:
    if "```" in text:
        for p in text.split("```")[1:]:
            body = p
            for tag in ("regex", "python", "re", "text"):
                if body.lower().startswith(tag):
                    body = body[len(tag):]
                    break
            body = body.strip().splitlines()[0].strip() if body.strip() else ""
            if body:
                return body
        return None
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return line or None


def generate(n: int) -> None:
    from pipeline import chat

    def one(i: int) -> dict:
        try:
            t = chat([{"role": "user", "content": PROMPT}],
                     temperature=1.0, max_tokens=180)
        except Exception as e:
            return {"i": i, "error": str(e)[:200]}
        p = extract(t or "")
        if not p:
            return {"i": i, "error": "no pattern"}
        if p.startswith(("r'", 'r"')):
            p = p[2:-1] if p[-1] in "'\"" else p[2:]
        elif p[0] in "'\"" and p[-1] == p[0]:
            p = p[1:-1]
        if p.startswith("/") and p.rfind("/") > 0:
            p = p[1:p.rfind("/")]
        return {"i": i, "pattern": p}

    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(one, range(n)))
    with (OUT / "corpus.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"generated {len(rows)}, with pattern "
          f"{sum(1 for r in rows if r.get('pattern'))}")


RUNNER = r'''
import json, re, sys
pat, probes = json.loads(sys.stdin.read())
try:
    rx = re.compile(pat)
except Exception as e:
    print(json.dumps({"error": type(e).__name__})); sys.exit(0)
hits = []
for i, s in enumerate(probes):
    try:
        if rx.search(s):
            hits.append(i)
    except Exception:
        pass
print(json.dumps({"hits": hits}))
'''


def run_one(pat: str, timeout: float = 10.0) -> list[int] | None:
    try:
        p = subprocess.run([sys.executable, "-c", RUNNER],
                           input=json.dumps([pat, PROBES]),
                           capture_output=True, text=True, timeout=timeout)
        d = json.loads(p.stdout.strip().splitlines()[-1])
        return d.get("hits")
    except Exception:
        return None


def execute() -> None:
    rows = [json.loads(l) for l in (OUT / "corpus.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("pattern")]
    with ThreadPoolExecutor(max_workers=8) as ex:
        hits = list(ex.map(lambda r: run_one(r["pattern"]), rows))
    # an expression matching nothing, or everything, carries no discrimination
    keep = [{"pattern": r["pattern"], "hits": h}
            for r, h in zip(rows, hits)
            if h is not None and 0 < len(h) < len(PROBES)]
    with (OUT / "executed.jsonl").open("w") as fh:
        for k in keep:
            fh.write(json.dumps(k) + "\n")
    print(f"compiled+ran {len(rows)}, discriminating {len(keep)} "
          f"(probe corpus {len(PROBES)})")


def language_distance(hits: list[list[int]]) -> np.ndarray:
    sets = [set(h) for h in hits]
    n = len(sets)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            u = len(a | b)
            D[i, j] = D[j, i] = 1.0 - (len(a & b) / u if u else 1.0)
    return D


def match_coverage(hits: list[list[int]]):
    sets = [set(h) for h in hits]

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
        if r["pattern"] not in seen:
            seen.add(r["pattern"])
            uniq.append(r)
    rows = uniq
    pats = [r["pattern"] for r in rows]
    hits = [r["hits"] for r in rows]
    print(f"{len(rows)} distinct patterns; "
          f"{len({tuple(h) for h in hits})} distinct languages")

    E = embed(pats)
    TD = P.pairwise_cosine(E)
    AD = language_distance(hits)
    prof = P.near_field_profile(None, AD, text_dist=TD, n_bins=n_bins)
    P.print_profile(prof, "regex -> accepted language")

    dd = P.dedup_quality(TD, AD)
    print(f"\n  same-language pairs {dd['n_duplicate_pairs']} "
          f"({100*dd['duplicate_pair_rate']:.2f}%); AUC {dd['auc']}")
    if dd["best"]:
        b = dd["best"]
        print(f"  best radius t={b['t']:.3f}  P {b['precision']:.3f} "
              f"R {b['recall']:.3f}  F1 {b['f1']:.3f}")

    cov = match_coverage(hits)
    budgets = [b for b in (10, 20, 30, 50, 100) if b < len(rows)]
    sel = P.selector_comparison(TD, cov, budgets, n_seeds=20)
    names = (["random", "maxmin"]
             + [a for a in sel["arms"] if a.startswith("filter@")]
             + (["oracle"] if "oracle" in sel["arms"] else []))
    print("\n  probe strings matched (mean of 20 seeds)")
    print("   k     " + "".join(f"{a:>16}" for a in names))
    for k in budgets:
        print(f"  {k:>3}    " + "".join(f"{sel['arms'][a][k]['mean']:>16.1f}" for a in names))

    pur = P.reject_purity(TD, AD)
    _print_purity(pur)
    rep = P.report(prof, dd, sel, "regex->language", pur)
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
