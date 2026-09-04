# Proxy Embeddings: Steering and Measuring Diversity Through a Space That Is Not the Artifact's

Site: https://thehalleyyoung.github.io/proxy-embeddings/ · Paper: [`paper/paper.pdf`](paper/paper.pdf)
· Main paper: [Recursive Axis Conditioning](https://thehalleyyoung.github.io/rac/)

Every diversity objective in this line of work is computed in an embedding, and
the embedding is a proxy: for a rendered image it is a text encoder's view of the
instruction, for a poem it is a semantic model nearly blind to metre, for an exam
item it is a space in which the answer key does not exist. [Recursive Axis
Conditioning](https://thehalleyyoung.github.io/rac/) assumes an embedding oracle
and no inverse. This paper asks what follows when the oracle is a proxy for the
artifact rather than the artifact itself, and extends the method across that
gap: measuring the gap, steering through it, auditing conditioning at the seam
where one model's language becomes another model's input, and balancing by
construction what no embedding encodes.

## Headline measurements

| finding | number |
|---|---|
| text-embedding vs rendered-image pairwise similarity, seven arms | Pearson ***r* = 0.167–0.421**; 14.5% shared variance pooled, 8.2% within-arm |
| prompt-to-track alignment, naive arm | MuQ-MuLan **0.68**, CLAP-fused 0.50, CLAP-music **0.18** |
| cross-embedder agreement on track similarity | 0.22–0.84; per-arm verdicts flip |
| image axes realized in the written instruction / in the render | **9 of 11 / 4 of 11**, mean loss 35% (Wilcoxon *p* = .005) |
| generated exam bank, share of keys in position A | **90%** (χ² = 90.4); after uniform permutation 2.6 |
| vision-steered arm vs text-only RAC, centered Vendi at *n* = 200 | **91.97 vs 77.67** (+18%), best arm in the domain |
| cross-modal audio steering, CLAP Vendi on rendered instrumentals | **17.43** vs 11.5 naive, 14.1 axis-conditioned |
| best-of-3 renders selected on pixel statistics | optical min-gap **1.24×**, CLIP min-gap −0.001 |
| ranking under four independent representations | RAC **1st under all four**, mean Kendall τ +0.83 |
| structural repetition (layout twins), baselines vs our arms | 0.223 vs 0.473 — the one channel where baselines win |
| literal-space structural bans, palette twins / coverage objective | 0.78 → 0.35 / +21% |

## What's here

```
paper/      source.md (the manuscript), build_paper.py, figures/, and the built
            paper.md, paper.tex, paper.pdf
index.html  the GitHub Pages site: the paper with every figure embedded
build_site.py, site.css, preview.sh
code/       the scripts behind every number here (curated from the main repo)
data/       result JSONs and the provenance registry
```

    python3 paper/build_paper.py     # paper.md, paper.tex, paper.pdf
    python3 build_site.py            # index.html from paper/paper.md
    ./preview.sh                     # open the site locally

## Reproducing

| section | script |
|---|---|
| §4.2 text vs image proxy gap, contact sheet | `code/vision_compare.py`, `code/render_images.py`, `code/make_contact_sheet.py` |
| §4.3 audio embedder dependence | `code/audio_embedders.py`, `code/audio_compare.py`, `code/audio_domain.py` |
| §4.4 ranking under four representations | `code/rac_improve/is_diversity_identified.py`, `identifiability_v2.py` |
| §4.5 literal vs latent | `code/metrics.py` |
| §4.6 structural repetition and bans | `code/structural.py`, `code/image_steer5.py`–`image_steer7.py` |
| §4.7 key position | `code/rac_improve/item_keybias.py`, `balance_keys.py` |
| §5.1 vision-steered arm | `code/image_steer.py`, `code/vision_loop.py` |
| §5.2 cross-modal audio steering | `code/lyria_steer.py`, `lyria_calibrate.py`, `lyria_online.py` |
| §5.3 best-of-*K* at the render | `code/rac_improve/render_best_of_k.py`, `optical_consistent.py` |
| §6 the seam, by intervention | `code/rac_improve/probe_image_seam.py`, `axis_realization.py`, `text_realization.py`, `item_realization.py` |
| §7 artifact-space level vectors | `code/calculus.py` (`score_axes_realized`) |

Live runs need `OPENROUTER_API_KEY`, a local Ollama with `nomic-embed-text`,
`OPENAI_API_KEY` for images and a Gemini key for Lyria audio. Rendered images
carry no logged seed because the image endpoint exposes none; see the paper's
§9.

## Licence

Code MIT. Paper CC BY 4.0.
