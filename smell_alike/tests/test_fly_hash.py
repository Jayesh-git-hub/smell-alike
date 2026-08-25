"""
test_fly_hash.py
================
Phase 1 sanity tests for FlyHash.
All three must pass before moving to Phase 2.

Run with:  pytest smell_alike/tests/test_fly_hash.py -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from smell_alike.fly_hash import FlyHash


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def hasher():
    return FlyHash(input_dim=128, kenyon_cells=2000, sparsity=0.05, seed=42)


def _rand_unit(dim, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


# ── Test 1: Determinism ───────────────────────────────────────────────────────

def test_same_input_same_hash(hasher):
    """Same input always produces the same fingerprint."""
    x = _rand_unit(128, seed=1)
    fp1 = hasher.hash(x)
    fp2 = hasher.hash(x)
    np.testing.assert_array_equal(fp1, fp2, err_msg="Hash is not deterministic!")


# ── Test 2: Sparsity ──────────────────────────────────────────────────────────

def test_fingerprint_sparsity(hasher):
    """Each fingerprint should have ~5% bits set (within reasonable tolerance)."""
    rng = np.random.default_rng(99)
    X = rng.standard_normal((200, 128))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    fps = hasher.hash_batch(X)

    bit_counts = fps.sum(axis=1)
    mean_bits = bit_counts.mean()
    target = hasher.k  # should be ~100 out of 2000

    print(f"\n  Target bits set : {target}")
    print(f"  Mean bits set   : {mean_bits:.1f}")
    print(f"  Actual sparsity : {mean_bits/2000:.4f}")

    assert abs(mean_bits - target) <= 5, (
        f"Sparsity off: mean {mean_bits:.1f} bits, expected ~{target}"
    )


# ── Test 3: Similarity property ───────────────────────────────────────────────

def test_similar_inputs_higher_overlap(hasher):
    """
    Similar inputs should have higher Jaccard overlap than random pairs.
    We create a 'near-duplicate' by adding tiny noise.
    """
    x = _rand_unit(128, seed=7)

    # Near-duplicate: x + tiny noise, then re-normalise
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(128) * 0.05
    x_near = x + noise
    x_near /= np.linalg.norm(x_near)

    # Random unrelated vector
    x_rand = _rand_unit(128, seed=999)

    fp_x     = hasher.hash(x)
    fp_near  = hasher.hash(x_near)
    fp_rand  = hasher.hash(x_rand)

    sim_near = hasher.similarity(fp_x, fp_near)
    sim_rand = hasher.similarity(fp_x, fp_rand)

    print(f"\n  Jaccard(x, near-dup) = {sim_near:.4f}")
    print(f"  Jaccard(x, random)   = {sim_rand:.4f}")

    assert sim_near > sim_rand, (
        f"Near-duplicate ({sim_near:.4f}) should score higher than random ({sim_rand:.4f})"
    )


# ── Test 4: find_neighbors returns correct type + count ───────────────────────

def test_find_neighbors_shape(hasher):
    """find_neighbors returns the right number of results."""
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((50, 128))
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus_fps = hasher.hash_batch(corpus)

    query = _rand_unit(128, seed=5)
    idx, scores = hasher.find_neighbors(query, corpus_fps, top_k=5)

    assert len(idx) == 5
    assert len(scores) == 5
    assert (scores[:-1] >= scores[1:]).all(), "Scores should be in descending order"


# ── Test 5: Batch equals row-by-row ──────────────────────────────────────────

def test_hash_batch_consistent(hasher):
    """hash_batch must produce the same result as calling hash() per row."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((20, 128))
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    batch_fps = hasher.hash_batch(X)
    row_fps   = np.array([hasher.hash(x) for x in X])
    np.testing.assert_array_equal(batch_fps, row_fps)


# ── Test 6: Wrong input dim raises ───────────────────────────────────────────

def test_wrong_dim_raises(hasher):
    with pytest.raises(ValueError, match="Expected 128"):
        hasher.hash(np.ones(64))


# ── Test 7: Sparsity report ───────────────────────────────────────────────────

def test_sparsity_report(hasher):
    rng = np.random.default_rng(3)
    X = rng.standard_normal((100, 128))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    fps = hasher.hash_batch(X)
    report = hasher.sparsity_report(fps)

    print(f"\n  Sparsity report: {report}")
    assert "actual_sparsity" in report
    assert 0.03 < report["actual_sparsity"] < 0.07, "Actual sparsity far from 5%"


# ── Test 8: repr ──────────────────────────────────────────────────────────────

def test_repr(hasher):
    r = repr(hasher)
    assert "FlyHash" in r
    assert "2000" in r
