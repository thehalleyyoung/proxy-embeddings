"""Offline analysis of probe_generations.jsonl -- no API calls.

Separated from the probe itself so that a bug in the analysis never costs a
second round of generations.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from probe_contract_load import effect, N_PERM  # noqa: E402

recs = [json.loads(l) for l in (HERE / "probe_generations.jsonl").read_text().splitlines() if l.strip()]
ok = [r for r in recs if r.get("text")]
# A cell with k levels and barely more than k points has eta^2 near 1 by
# construction, and the null correction (k-1)/(n-1) is then subtracting almost
# everything -- the estimate is all variance. Require 3 points per level.
MIN_PER_LEVEL = 3
print(f"{len(ok)}/{len(recs)} generations with text")
print(f"cells need >= {MIN_PER_LEVEL} generations per level; smaller cells are dropped\n")
print(f"{'axis':<34}{'n':>5}{'rho@7':>9}{'rho@11':>9}{'delta':>9}{'rho(all)':>10}{'p':>8}")
rows = []
for ax in sorted({r['axis'] for r in ok}):
    sel = [r for r in ok if r['axis'] == ax]
    cell = {}
    for load in (7, 11):
        s = [r for r in sel if r['load'] == load]
        k = len({r['level'] for r in s})
        cell[load] = (effect([r['text'] for r in s], [r['level'] for r in s],
                             [r['base'] for r in s])[0]
                      if k >= 2 and len(s) >= MIN_PER_LEVEL * k else float('nan'))
    kk = len({r['level'] for r in sel})
    if len(sel) < MIN_PER_LEVEL * kk:
        print(f"{ax[:33]:<34}{len(sel):>5}   (dropped: {len(sel)} gens for {kk} levels)")
        continue
    rho, p = effect([r['text'] for r in sel], [r['level'] for r in sel],
                    [(r['load'], r['base']) for r in sel], n_perm=N_PERM)
    d = cell[7] - cell[11]
    rows.append({"axis": ax, "n": len(sel), "rho_7": cell[7], "rho_11": cell[11],
                 "delta": d, "rho_all": rho, "p": p})
    print(f"{ax[:33]:<34}{len(sel):>5}{cell[7]:>9.4f}{cell[11]:>9.4f}{d:>+9.4f}{rho:>10.4f}{p:>8.4f}")
g = [r for r in rows if np.isfinite(r['delta'])]
print(f"\n{'MEAN':<34}{'':>5}{np.nanmean([r['rho_7'] for r in g]):>9.4f}"
      f"{np.nanmean([r['rho_11'] for r in g]):>9.4f}"
      f"{np.nanmean([r['delta'] for r in g]):>+9.4f}"
      f"{np.nanmean([r['rho_all'] for r in rows]):>10.4f}")
print(f"{sum(1 for r in g if r['delta']>0)}/{len(g)} axes higher at load 7")
sig = [r for r in rows if r['p'] < 0.05]
print(f"axes realized under manipulation (p<0.05): {len(sig)}/{len(rows)}")
json.dump(rows, open(HERE / "probe_contract_load.json", "w"), indent=2)
