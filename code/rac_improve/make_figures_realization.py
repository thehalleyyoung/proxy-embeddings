"""Figures 16-19: axis realization, the scoring inversion, and form uniformity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
FIG = RESEARCH / "figures"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(HERE))

plt.rcParams.update({
    "figure.dpi": 220, "savefig.dpi": 220, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})

INK = "#222222"
HOT = "#d62728"
COOL = "#1f77b4"
WARM = "#ff7f0e"
GREY = "#999999"


# ---------------------------------------------------------------- fig 16
def fig16_contact_sheet():
    """Eight max-min renders, as read in the diagnosis."""
    from PIL import Image
    src = RESEARCH / "real" / "dalle_steer2_maxmin" / "images"
    idx = [0, 8, 16, 24, 32, 40, 48, 56]
    paths = [src / f"{i:05d}.png" for i in idx]
    if not all(p.exists() for p in paths):
        print("fig16: images missing, skipped")
        return
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.6))
    for ax, p, i in zip(axes.ravel(), paths, idx):
        ax.imshow(Image.open(p).convert("RGB").resize((420, 420)))
        ax.set_title(f"#{i}", fontsize=8, color=INK, pad=2)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Eight of sixty max–min renders, evenly spaced through the run",
                 fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "fig16_maxmin_sample.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig16_maxmin_sample.png")


# ---------------------------------------------------------------- fig 17
def fig17_realization():
    """Per-axis realization: images (mostly inert) vs poems (all realized)."""
    from axis_realization import audit_run
    from text_realization import audit_corpus

    img = audit_run(RESEARCH / "real" / "dalle_steer2_maxmin")
    txt = audit_corpus(RESEARCH / "live" / "poems")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # -- images: CLIP identification lift over chance
    a = sorted(img["axes"], key=lambda r: r["clip_id_lift"])
    names = [r["axis"][:26] for r in a]
    lift = [r["clip_id_lift"] for r in a]
    sig = [r["clip_id_p"] < 0.05 or r["literal_eta2_p"] < 0.05 for r in a]
    axes[0].barh(names, lift, color=[COOL if s else HOT for s in sig], height=0.62)
    axes[0].axvline(1.0, color=INK, lw=1.2, ls="--")
    axes[0].set_xlabel("CLIP identification of the commanded level  (1.0 = chance)")
    axes[0].set_title("Images: the commanded level is often undetectable\n"
                      f"(n = {img['n_items']} renders, 7 axes)", fontsize=9.5)
    axes[0].text(1.01, len(names) - 0.35, "chance", fontsize=7.5, color=INK,
                 rotation=90, va="top")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[
        Patch(color=COOL, label="detectable in the render (p < .05)"),
        Patch(color=HOT, label="undetectable on both channels")],
        loc="lower right", fontsize=7.5)

    # -- poems: semantic + prosodic eta^2
    b = sorted(txt["axes"], key=lambda r: r["semantic_eta2"])
    nm = [r["axis"][:26] for r in b]
    y = np.arange(len(nm))
    axes[1].barh(y - 0.19, [r["semantic_eta2"] for r in b], height=0.36,
                 color=COOL, label="semantic (nomic)")
    axes[1].barh(y + 0.19, [r["prosodic_eta2"] for r in b], height=0.36,
                 color=WARM, label="prosodic (form)")
    null = 4 / (txt["n_items"] - 1)
    axes[1].axvline(null, color=INK, lw=1.2, ls="--")
    axes[1].text(null * 1.05, -0.8, "null", fontsize=7.5, color=INK)
    axes[1].set_yticks(y); axes[1].set_yticklabels(nm)
    axes[1].set_xlabel(r"$\eta^2$: share of corpus variance explained by the axis")
    axes[1].set_title("Poems: every axis is realized\n"
                      f"(n = {txt['n_items']} poems, 7 axes)", fontsize=9.5)
    axes[1].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG / "fig17_axis_realization.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig17_axis_realization.png")


# ---------------------------------------------------------------- fig 18
def fig18_scoring_inversion():
    """Ground truth: attribution ranks the inert axis first; manipulation does not."""
    from calculus import score_axes_realized
    from calculus2 import axis_realization, score_axes_realized_causal
    from test_scoring_ground_truth import make_corpus

    trials = 24
    old_pos, new_pos = {"A": [], "B": [], "C": []}, {"A": [], "B": [], "C": []}
    rho_all = {"A": [], "B": [], "C": []}
    for seed in range(trials):
        specs, E = make_corpus(seed)
        o = [s.name for s in score_axes_realized(specs, E)]
        n_ = [s.name for s in score_axes_realized_causal(specs, E)]
        r = axis_realization(specs, E)
        for k in ("A", "B", "C"):
            old_pos[k].append(o.index(k) + 1)
            new_pos[k].append(n_.index(k) + 1)
            rho_all[k].append(r.get(k, 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    labels = ["A\nstrong effect", "C\nweak effect", "B\nno effect"]
    order = ["A", "C", "B"]
    x = np.arange(3)
    om = [np.mean(old_pos[k]) for k in order]
    nm = [np.mean(new_pos[k]) for k in order]
    oe = [np.std(old_pos[k]) for k in order]
    ne = [np.std(new_pos[k]) for k in order]
    axes[0].bar(x - 0.19, om, 0.36, yerr=oe, capsize=3, color=HOT,
                label="attribution (§4.2)")
    axes[0].bar(x + 0.19, nm, 0.36, yerr=ne, capsize=3, color=COOL,
                label="manipulation (§4.4)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("mean rank assigned  (1 = most promising)")
    axes[0].set_ylim(0, 3.6)
    axes[0].invert_yaxis()
    axes[0].set_title(f"Ranking three axes whose effects are known\n"
                      f"by construction ({trials} seeds)", fontsize=9.5)
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].bar(x, [np.mean(rho_all[k]) for k in order],
                yerr=[np.std(rho_all[k]) for k in order], capsize=3,
                color=[COOL, WARM, GREY], width=0.55)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel(r"realization $\rho$")
    axes[1].set_title("The realization factor recovers the true ordering",
                      fontsize=9.5)

    fig.tight_layout()
    fig.savefig(FIG / "fig18_scoring_inversion.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig18_scoring_inversion.png")


# ---------------------------------------------------------------- fig 19
def fig19_form_uniformity():
    """What the poem corpus never varies."""
    from text_realization import audit_corpus
    txt = audit_corpus(RESEARCH / "live" / "poems")
    cf = txt["corpus_form"]
    want = ["type_token", "word_len", "n_lines", "line_chars_mean",
            "syl_per_line_mean", "comma_rate", "line_chars_sd",
            "rhyme_density", "syl_per_line_sd", "first_person"]
    items = [(k, cf[k]["cv"]) for k in want if k in cf]
    items.sort(key=lambda kv: kv[1])
    names = [k.replace("_", " ") for k, _ in items]
    cvs = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(names, cvs, color=[HOT if c < 0.3 else COOL for c in cvs], height=0.62)
    ax.axvline(0.3, color=INK, lw=1.1, ls="--")
    ax.set_xlabel("coefficient of variation across the 60-poem corpus")
    ax.set_title("Form dimensions the poem corpus barely moves\n"
                 "(low = every poem is the same on this axis)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig19_form_uniformity.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig19_form_uniformity.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig16_contact_sheet()
    fig17_realization()
    fig18_scoring_inversion()
    fig19_form_uniformity()
