"""Tests for decodergap, on synthetic decoders whose answers are known.

Each test fixes a decoder whose relationship to the text is known by
construction, so the tool's verdict can be checked against ground truth rather
than against another measurement.
"""
import numpy as np
import decodergap as dg


def _texts(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(list("abcdefghij"), 12)) for _ in range(n)]


def _emb(dim=24, seed=1):
    rng = np.random.default_rng(seed)
    table = {c: rng.normal(size=dim) for c in "abcdefghij"}

    def embed(texts):
        return np.array([np.sum([table[c] for c in t.split()], axis=0)
                         for t in texts])
    return embed


def test_pairwise_cosine_is_a_metric_on_itself():
    E = np.random.default_rng(0).normal(size=(20, 8))
    D = dg.pairwise_cosine(E)
    assert D.shape == (20, 20)
    assert np.allclose(np.diag(D), 0, atol=1e-9)
    assert np.allclose(D, D.T)
    assert (D >= -1e-9).all()


def test_completion_check_refuses_unequal_arms():
    equal = {"a": [True] * 90 + [False] * 10, "b": [True] * 88 + [False] * 12}
    assert dg.completion_check(equal)["ok"]
    unequal = {"a": [True] * 95 + [False] * 5, "b": [True] * 60 + [False] * 40}
    r = dg.completion_check(unequal)
    assert not r["ok"] and "REFUSED" in r["verdict"]
    assert abs(r["spread"] - 0.35) < 1e-9


def test_diagnose_misses_separates_the_two_failure_shapes():
    near = [{"present": True, "x": 1, "y": 0} for _ in range(50)]
    assert "UNOBSERVABLE" in dg.diagnose_misses(near, lambda m: m)["verdict"]
    absent = [{"present": False, "x": None, "y": None} for _ in range(50)]
    assert "EMERGENT" in dg.diagnose_misses(absent, lambda m: m)["verdict"]
    scattered = [{"present": True, "x": 4, "y": 5} for _ in range(50)]
    assert "EMERGENT" in dg.diagnose_misses(scattered, lambda m: m)["verdict"]


def test_greedy_oracle_beats_random_on_a_coverage_it_can_see():
    rng = np.random.default_rng(3)
    sets = [set(rng.choice(200, 12, replace=False)) for _ in range(80)]

    def cov(idx):
        s = set()
        for i in idx:
            s |= sets[i]
        return float(len(s))
    orc = cov(dg.greedy_oracle(cov, 80, 10, 0, 0))
    rnd = np.mean([cov(dg.random_pick(80, 10, s)) for s in range(20)])
    assert orc > rnd


def test_triage_flags_an_exact_emergent_target_and_suggests_a_band():
    r = dg.triage("the clip must last exactly 2.4 seconds")
    assert r.kind == "emergent" and r.exactness == "exact"
    assert r.suggestion and "range" in r.suggestion
    c = dg.triage("the image must contain a red door")
    assert c.kind == "constructible" and c.suggestion is None


def test_audit_reports_the_control_and_refuses_when_it_is_not_cleared():
    texts = _texts()
    embed = _emb()
    # artifact distance INDEPENDENT of the text: the control must not be cleared
    rng = np.random.default_rng(7)
    A = rng.random((len(texts), len(texts)))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0.0)
    rep = dg.audit(texts, embed, artifact_distance=A, coverage=None)
    assert "baselines" in rep.control
    assert not rep.control["clears"]
    assert rep.verdicts["correlational_validation"]["verdict"] == "CONFOUNDED"
    assert "NO" in rep.summary()


def test_audit_runs_with_a_real_decode_function():
    texts = _texts(60)
    embed = _emb()

    def decode(t):
        return {c for c in t.split()}

    def dist(a, b):
        return 1.0 - len(a & b) / max(len(a | b), 1)

    rep = dg.audit(texts, embed, decode=decode, distance=dist, coverage=None)
    assert rep.profile.n_items == 60
    assert len(rep.profile.deciles) == 10
