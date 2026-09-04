"""decodergap — what a text channel can and cannot tell you about the artifact.

You address a generator in one modality and need the output in another: you
write text, a decoder you did not train turns it into a program, a query, an
image, a video. This package measures what your text-space machinery is buying,
and what to do instead.

    import decodergap as dg

    dg.audit(texts, embed, decode, distance, coverage)  # is my distance machinery sound?
    dg.triage("the shot must be exactly 2.4 seconds")   # how should I state this target?
    dg.plan(decode_budget=2000, pool=5000, select=50)   # what does my decode budget buy?
    dg.completion_check(arms)                           # did my arms answer equally often?
    dg.diagnose_misses(misses, axis_distance)           # which axis am I failing on?

Two of these exist because they caught errors in the work that produced them:
`completion_check` found a 19-point effect that was an unequal generation budget,
and `diagnose_misses` found that a failed pre-registration was measuring the
authors' own quantization rather than the property they meant to test.

Everything is domain-agnostic. Supply a decode function returning anything with a
distance on it and every number follows.
"""

from ._api import (
    audit,
    Audit,
    triage,
    Triage,
    plan,
    completion_check,
    diagnose_misses,
)
from ._core import (
    pairwise_cosine,
    near_field_profile,
    greedy_maxmin,
    greedy_oracle,
    filter_then_sample,
    random_pick,
    selector_comparison,
    dedup_quality,
    reject_purity,
)

__version__ = "0.1.0"

__all__ = [
    "audit", "Audit", "triage", "Triage", "plan",
    "completion_check", "diagnose_misses",
    "pairwise_cosine", "near_field_profile", "greedy_maxmin", "greedy_oracle",
    "filter_then_sample", "random_pick", "selector_comparison",
    "dedup_quality", "reject_purity",
]
