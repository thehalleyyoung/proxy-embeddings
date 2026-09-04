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
