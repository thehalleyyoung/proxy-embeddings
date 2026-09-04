# Pre-registration: constructible vs emergent, on the SVG decoder

Written **before** the SVG lever run, after the rule was confirmed retrospectively
on SQL. Recorded here so the prediction cannot be adjusted to the outcome.

## The rule (fixed in advance, mechanical)

For a target property of a decoded artifact, ask:

> Does satisfying it require **writing a specific token into the artifact**, or
> **finding a computation whose behaviour lands somewhere**?

The first is CONSTRUCTIBLE, the second EMERGENT.

Applied retrospectively to SQL, using base-schema membership of the target column
as the mechanical criterion (a base column can be named in a `WHERE` clause; an
alias for an aggregate cannot), the rule gave:

    constructible  instruct 79.0%  [70.4, 87.7]  n=81
    emergent       instruct 31.3%  [22.2, 40.4]  n=99
    difference     +47.7 points, 95% CI [+34.8, +60.2]

with rarity retaining a smaller effect within each kind (+0.251, +0.252).

## Prediction for SVG, registered before the run

The SVG artifact is the set of marks drawn: for each element, its tag, its
position on a 6x6 grid, a size band and a fill-hue band. A target is one such
mark.

**Every SVG mark target is CONSTRUCTIBLE by the rule**: a required
`circle@2,3|s1|h4` is satisfied by writing a `<circle>` with those coordinates
and that fill. Nothing has to emerge from a computation.

Therefore we predict:

1. `instruct` compliance on SVG will be **high (>= 70%)** and much closer to the
   regular-expression decoder (97.8%) than to the program decoder (62.5%).
2. Compliance will be **approximately flat in rarity**, including on the
   never-produced band, with no monotone decay of the kind the program decoder
   shows (100% -> 25%).
3. `decoy` will remain near the floor, well below `instruct`.

**Falsifier.** If SVG `instruct` compliance decays with rarity in the way
programs do, or lands below 50% overall, the constructible/emergent distinction
does not predict reach and should be reported as unsupported rather than
rescued.

A second, weaker prediction for the same run: because SVG marks are quantized,
some misses will be near-misses (right tag, adjacent grid cell). Those count as
misses here; a graded score is not used, because the whole point of the
compliance measure is that it is checked by decoding rather than judged.

## Addendum, added before the SVG run: SVG is a positive control

A collaborating session pointed out that this is better described as a positive
control than as a treatment. Every SVG mark target is constructible *and*
observable — the generator is told the exact tag, grid cell, size band and hue
band, and can write an element with those properties directly, needing no
information it does not have. So if compliance is **not** high here, the fault is
in the rule rather than in the domain, and a positive control that fails is more
informative than a treatment that succeeds.

## Second addendum: emergence and observability are separate axes

The same session found, on a glob-pattern decoder, constructible targets at
100.0% and emergent targets at exactly 0.0% — and flagged that its emergent task
may have been *impossible* rather than merely emergent, because the generator was
shown 12 sample paths and asked to match a count over a 600-name corpus it could
not see. Our SQL emergent targets scored 31.3%, which is plainly reachable.

So two axes, not one:

- **emergent**: satisfying the target requires finding a computation whose
  behaviour lands somewhere, rather than writing a token.
- **unobservable**: the generator cannot see the space the behaviour is scored
  against, so it cannot check its own work even in principle.

Our SQL emergent targets are emergent-but-partly-observable (the schema is in the
prompt, the data is not). A practitioner steering a closed-weight video model is
usually in the emergent-and-unobservable regime, which is the worse one. We
predict compliance orders: constructible > emergent-observable >
emergent-unobservable, and we record here that the SQL number (31.3%) is an
estimate for the middle case and should not be quoted for the third.

---

## Outcome: the SVG prediction FAILED its own falsifier

Recorded here rather than adjusted. 52 targets × 3 arms × 3 seeds.

| prediction | registered | observed | verdict |
|---|---|---|---|
| 1. `instruct` ≥ 70% | ≥ 70% | **18.6%** [12.2, 25.0] | **FAILED** |
| 2. approximately flat in rarity | flat | 13.9 / 11.1 / 30.6 / 16.7 / 25.0 — not flat, not monotone | **FAILED** |
| 3. `decoy` near the floor, below `instruct` | yes | 3.2% vs 18.6% | held |

The registered falsifier reads: *if SVG `instruct` compliance decays with rarity
in the way programs do, or lands below 50% overall, the constructible/emergent
distinction does not predict reach and should be reported as unsupported rather
than rescued.* It landed at 18.6%. **On this test the distinction is not
supported**, and the paper says so.

### The diagnosis, which is a separate and clearly post-hoc claim

The most likely reason SVG behaves this way is that its target is not the simple
constructible thing the registration assumed. Every other decoder's target is a
single property. An SVG target is a **conjunction of four quantized properties** —
tag, grid cell (1 of 36), size band (1 of 5), fill-hue band (1 of 9) — and three
of those quantizations are *our measurement apparatus*, not anything the
generator can compute. It is told "column 5 of 6, row 2 of 6, medium in area,
blue"; it cannot check whether the circle it drew has an area fraction above the
0.05 threshold that separates our size bands, or whether its chosen blue falls in
hue bin 5 or 6.

So SVG is constructible **in principle** and unobservable **in practice**: the
generator cannot verify its own compliance. That is the observability axis, and
it means the registration mis-classified the domain rather than the rule
necessarily being wrong.

That explanation is post-hoc and is not evidence. It makes one testable
prediction, which is being run: **if the misses are quantization misses, scoring
the same generations with one band of tolerance on cell and size should raise
compliance sharply; if the misses are wholesale, it should not.** The re-run
records the marks each generation actually produced so the miss distance can be
measured rather than assumed. Whatever that returns is reported next to this
failure, not in place of it.

---

## Registered before scoring the lever re-run at the equalized budget

The lever result — the paper's headline — was originally measured at a 900-token
generation budget under which the arms did not complete equally: `instruct`
returned a usable artifact 63.2% of the time against `blind`'s 99.3%, a 36-point
spread. Intent-to-treat figures compared across arms under that spread are not
comparable, so both lever experiments are re-run at 2,600 tokens with one retry,
into new directories, leaving the originals auditable.

Three outcomes, registered before the data are read:

1. **ITT grows, conditional roughly unchanged.** The expected result. Attrition
   ran against the instructed arm, so equalizing completion should widen the
   intent-to-treat gap while leaving the conditional figures (98.9% programs,
   79.2% SQL) near where they were, since those already conditioned on answering.

2. **ITT and conditional both roughly unchanged.** Budget was not the binding
   constraint on the lever, and the original figures stand as reported.

3. **ITT grows and conditional FALLS.** Named by a collaborating session and not
   by us. This would mean the extra budget lets *marginal* attempts through —
   generations that previously ran out while struggling and now emit a wrong
   answer — converting non-returns into misses, which raises the denominator of
   the conditional figure without raising its numerator. If we see this, the
   original 98.9% was partly **survivorship**: only the confident attempts got
   far enough to emit. That would not undermine the lever, but it would mean the
   conditional number — the one a practitioner uses when they can retry —
   overpromises, and we would report the equalized conditional figure as the
   honest one and say why it fell.

The falsifier for the lever itself is unchanged and separate: if `instruct` does
not exceed `decoy` by an interval-excluding margin at the equalized budget, on
both decoders, the lever result does not survive and the paper's positive half
goes with it.
