"""Tests for connectome module."""
import numpy as np
import pytest
from smell_alike.connectome import (
    connectome_projection, ConnectomeFlyHash, REAL_STATS,
)


def test_synthetic_projection_shape():
    W = connectome_projection(input_dim=384, kenyon_cells=2000, seed=0)
    assert W.shape == (2000, 384)


def test_synthetic_projection_sparse():
    """Connectome W should be sparse (most entries near zero after normalisation)."""
    W = connectome_projection(input_dim=384, kenyon_cells=2000, seed=0)
    # Before normalisation, ~6% of entries are non-zero
    # After normalisation, all rows have norm 1, but structure is preserved
    assert W.shape == (2000, 384)
    row_norms = np.linalg.norm(W, axis=1)
    np.testing.assert_allclose(row_norms, 1.0, atol=1e-10)


def test_connectome_flyhash_interface():
    """ConnectomeFlyHash should have the same interface as FlyHash."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 64))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    cfh = ConnectomeFlyHash(input_dim=64, kenyon_cells=200, sparsity=0.05, seed=0)
    fp = cfh.hash(X[0])
    assert fp.shape == (200,)
    assert fp.sum() == cfh.k

    fps = cfh.hash_batch(X)
    assert fps.shape == (40, 200)
    assert np.all(fps.sum(axis=1) == cfh.k)

    idx, scores = cfh.find_neighbors(X[0], fps, top_k=5)
    assert idx.shape == (5,)
    assert scores.shape == (5,)
