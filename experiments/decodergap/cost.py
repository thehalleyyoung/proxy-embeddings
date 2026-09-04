"""What it costs to decode an item, against what it costs to embed one.

The prescription that follows from the near-field law — if you want coverage of
what the decoder does, measure what the decoder does — sounds expensive, and for
the decoders in this study it is not. This script measures both sides on the
same machine so the comparison is not an assumption.

Embedding is timed against a local Ollama server, which is the *favourable*
case for the text-space route: no network beyond loopback and no per-token
charge. A hosted embedding endpoint is slower and costs money.

    python3 cost.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "code"))


def timeit(fn, n: int) -> float:
    """Milliseconds per item."""
    t0 = time.perf_counter()
    fn()
    return 1000.0 * (time.perf_counter() - t0) / max(n, 1)


def main() -> None:
    out: dict = {}

    # ---- decode: python programs on the battery
    try:
        import domain_code as C
        rows = [json.loads(l) for l in
                (C.OUT / "executed.jsonl").read_text().splitlines()][:40]
        srcs = [r["src"] for r in rows]
        out["code_decode_ms"] = timeit(
            lambda: [C.run_one(s) for s in srcs], len(srcs))
        out["code_battery"] = len(C.BATTERY)
    except Exception as e:
        out["code_decode_error"] = str(e)[:120]

    # ---- decode: sql against the fixed database
    try:
        import domain_sql as S
        if S.DB.exists():
            rows = [json.loads(l) for l in
                    (S.OUT / "executed.jsonl").read_text().splitlines()][:200]
            qs = [r["sql"] for r in rows]
            out["sql_decode_ms"] = timeit(
                lambda: [S.run_query(q) for q in qs], len(qs))
    except Exception as e:
        out["sql_decode_error"] = str(e)[:120]

    # ---- decode: regex against the probe corpus
    try:
        import re as _re
        import domain_regex as R
        rows = [json.loads(l) for l in
                (R.OUT / "executed.jsonl").read_text().splitlines()][:200]
        pats = [r["pattern"] for r in rows]

        def match_all():
            for p in pats:
                try:
                    rx = _re.compile(p)
                    [rx.search(s) for s in R.PROBES]
                except Exception:
                    pass
        out["regex_decode_ms"] = timeit(match_all, len(pats))
        out["regex_probes"] = len(R.PROBES)
    except Exception as e:
        out["regex_decode_error"] = str(e)[:120]

    # ---- embed: local ollama, the favourable case for the text-space route
    try:
        from pipeline import embed
        import domain_code as C
        rows = [json.loads(l) for l in
                (C.OUT / "executed.jsonl").read_text().splitlines()][:64]
        texts = [r["src"] for r in rows]
        embed(texts[:8])  # warm the model
        out["embed_ms_batched64"] = timeit(lambda: embed(texts), len(texts))
        out["embed_ms_single"] = timeit(lambda: [embed([t]) for t in texts[:16]], 16)
    except Exception as e:
        out["embed_error"] = str(e)[:120]

    (HERE / "runs" / "cost.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    dec = {k.replace("_decode_ms", ""): v for k, v in out.items()
           if k.endswith("_decode_ms")}
    emb = out.get("embed_ms_batched64")
    if emb:
        print("\n  decoder                ms/item     x cheaper than embedding")
        for k, v in dec.items():
            print(f"  {k:<20} {v:9.3f}        {emb/v:9.1f}x")
        print(f"  {'embed (local, batched)':<20} {emb:9.3f}")
        if out.get("embed_ms_single"):
            print(f"  {'embed (local, single)':<20} {out['embed_ms_single']:9.3f}")


if __name__ == "__main__":
    main()
