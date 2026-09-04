"""Figures for the lever result and the decode budget.

Two panels each, drawn from the run JSONs rather than typed, so a figure cannot
disagree with the text.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE.parent.parent / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})

C = {"blind": "#9aa0a6", "decoy": "#c1443c", "instruct": "#1f5fa9"}


def load(f):
    return [json.loads(l) for l in (HERE / f).read_text().splitlines()]


def ci(h):
    h = np.array(h, dtype=float)
    if not h.size:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(0)
    bs = [rng.choice(h, h.size).mean() for _ in range(2000)]
    return h.mean(), np.percentile(bs, [2.5, 97.5])


def lever_figure():
    doms = [("programs", "runs/steer2/results.jsonl"),
            ("SQL", "runs/lever2_sql/results.jsonl"),
            ("regex", "runs/lever_regex/results.jsonl")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    arms = ["blind", "decoy", "instruct"]
    w, xs = 0.26, np.arange(len(doms))
    for k, arm in enumerate(arms):
        m, lo, hi = [], [], []
        for _, f in doms:
            rows = load(f)
            v, (l, h) = ci([bool(r.get("hit")) for r in rows if r["arm"] == arm])
            m.append(100 * v); lo.append(100 * (v - l)); hi.append(100 * (h - v))
        a1.bar(xs + (k - 1) * w, m, w, color=C[arm], label=arm,
               yerr=[lo, hi], capsize=2, error_kw={"lw": 0.7})
    a1.set_xticks(xs); a1.set_xticklabels([d for d, _ in doms])
    a1.set_ylabel("compliance, % of calls issued"); a1.set_ylim(0, 105)
    a1.legend(frameon=False, fontsize=7.5, loc="upper left")
    a1.set_title("A stated target is reached; a decoy in the\nsame words is not",
                 fontsize=8, loc="left")

    # rarity curve. The decoders do not share a band set -- the programs run
    # has no never-produced band -- so bands are aligned BY NAME rather than by
    # position, which is the bug the first version of this figure had.
    ORDER = ["[0,0.001) unobserved", "[0.001,0.02)", "[0.02,0.1)",
             "[0.1,0.3)", "[0.3,1.01)"]
    LABS = ["never\nproduced", "0.1-2%", "2-10%", "10-30%", ">30%"]
    for name, f, mk, col in (("programs", "runs/steer2/results.jsonl", "o", "#1f5fa9"),
                             ("SQL", "runs/lever2_sql/results.jsonl", "s", "#e07b39")):
        rows = load(f)
        have = {r["band"] for r in rows}
        xi, y, e, yb = [], [], [], []
        for i, b in enumerate(ORDER):
            if b not in have:
                continue
            v, (l, h) = ci([bool(r.get("hit")) for r in rows
                            if r["arm"] == "instruct" and r["band"] == b])
            xi.append(i); y.append(100 * v); e.append(100 * (h - l) / 2)
            yb.append(100 * ci([bool(r.get("hit")) for r in rows
                                if r["arm"] == "blind" and r["band"] == b])[0])
        a2.errorbar(xi, y, yerr=e, marker=mk, lw=1.4, ms=4, capsize=2,
                    color=col, label=f"{name}, instructed")
        a2.plot(xi, yb, marker=mk, ls=":", lw=1.0, ms=3, color="#9aa0a6",
                label=f"{name}, blind")
    a2.set_xticks(range(len(ORDER)))
    a2.set_xticklabels(LABS, fontsize=6.5)
    a2.set_xlabel("share of the natural corpus already hitting the target")
    a2.set_ylabel("compliance, %"); a2.set_ylim(-3, 108)
    a2.legend(frameon=False, fontsize=6.2, ncol=2, loc="lower right")
    a2.set_title("Reach extends past what the generator has ever produced",
                 fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_lever.png", bbox_inches="tight")
    print("wrote fig_lever.png")


def budget_figure():
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    for name, mk in (("code", "o"), ("sql", "s")):
        p = HERE / "runs" / f"budget_{name}.json"
        if not p.exists():
            continue
        b = json.loads(p.read_text())
        for row in b["rows"]:
            if row["k"] not in (10, 30):
                continue
            fr = sorted(float(f) for f in row["partial"])
            rec = [100 * (row["partial"][str(f)]["recovered"] or 0) for f in fr]
            ax.plot([100 * f for f in fr], rec, marker=mk, ms=3.5, lw=1.3,
                    label=f"{'programs' if name=='code' else 'SQL'}, select {row['k']}")
            mm = 100 * (row["maxmin"] - row["random"]) / (row["oracle"] - row["random"])
            ax.axhline(mm, ls=":", lw=0.8, color="#c1443c" if name == "code" else "#e0a030")
    ax.set_xlabel("% of the pool decoded")
    ax.set_ylabel("% of the oracle's advantage recovered")
    ax.set_ylim(-5, 105)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title("Dotted: what text-space selection buys", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_budget.png", bbox_inches="tight")
    print("wrote fig_budget.png")


if __name__ == "__main__":
    lever_figure()
    budget_figure()
