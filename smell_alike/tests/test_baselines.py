"""
test_baselines.py
=================
Sanity tests for ClassicLSH and PCAHash baselines.
Run with: pytest smell_alike/tests/ -v
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from smell_alike.baselines import ClassicLSH, PCAHash


def _corpus(n=100, dim=128, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


# ── ClassicLSH ────────────────────────────────────────────────────────────────

def test_classic_lsh_hash_shape():
    lsh = ClassicLSH(input_dim=128, n_bits=100)
    x = _corpus(1)[0]
    fp = lsh.hash(x)
    assert fp.shape == (100,)
    assert set(fp).issubset({0, 1})


def test_classic_lsh_batch():
    lsh = ClassicLSH(input_dim=128, n_bits=100)
    X = _corpus(50)
    fps = lsh.hash_batch(X)
    assert fps.shape == (50, 100)


def test_classic_lsh_neighbors():
    lsh = ClassicLSH(input_dim=128, n_bits=100)
    X = _corpus(50)
    fps = lsh.hash_batch(X)
    idx, scores = lsh.find_neighbors(X[0], fps, top_k=5)
    assert len(idx) == 5
    assert idx[0] == 0  # closest to itself


# ── PCAHash ───────────────────────────────────────────────────────────────────

def test_pca_hash_requires_fit():
    ph = PCAHash(n_bits=50)
    with pytest.raises(RuntimeError):
        ph.hash(np.ones(128))


def test_pca_hash_fit_and_hash():
    X = _corpus(200, 128)
    ph = PCAHash(n_bits=50)
    ph.fit(X)
    fp = ph.hash(X[0])
    assert fp.shape == (50,)
    assert set(fp).issubset({0, 1})


def test_pca_hash_batch():
    X = _corpus(200, 128)
    ph = PCAHash(n_bits=50).fit(X)
    fps = ph.hash_batch(X)
    assert fps.shape == (200, 50)
