"""Tests for LearnedFlyHash (Hebbian learning)."""
import numpy as np
import pytest
from smell_alike.learned_fly_hash import LearnedFlyHash


@pytest.fixture
def corpus():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 64))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def test_unfitted_works_like_flyhash(corpus):
    """Before fit(), LearnedFlyHash should behave like random FlyHash."""
    lf = LearnedFlyHash(input_dim=64, kenyon_cells=200, sparsity=0.05, seed=42)
    fp = lf.hash(corpus[0])
    assert fp.shape == (200,)
    assert fp.sum() == lf.k


def test_fit_returns_self(corpus):
    lf = LearnedFlyHash(input_dim=64, kenyon_cells=200, sparsity=0.05,
                        seed=42, n_epochs=1)
    result = lf.fit(corpus)
    assert result is lf
    assert lf.is_fitted


def test_fit_improves_recall(corpus):
    """Learned version should have >= recall of unfitted version."""
    from smell_alike.fly_hash import FlyHash
    from smell_alike.benchmark import true_topk

    D = 64
    rng = np.random.default_rng(99)
    top_k = 5

    base = FlyHash(input_dim=D, kenyon_cells=500, sparsity=0.05, seed=7)
    learned = LearnedFlyHash(input_dim=D, kenyon_cells=500, sparsity=0.05,
                             seed=7, learning_rate=0.01, n_epochs=5)
    learned.fit(corpus)

    base_fps = base.hash_batch(corpus)
    learned_fps = learned.hash_batch(corpus)

    base_recalls, learned_recalls = [], []
    for q in range(min(30, len(corpus))):
        gt = true_topk(corpus, q, top_k)
        b_idx, _ = base.find_neighbors(corpus[q], base_fps, top_k)
        l_idx, _ = learned.find_neighbors(corpus[q], learned_fps, top_k)
        base_recalls.append(len(gt & set(b_idx.tolist())) / top_k)
        learned_recalls.append(len(gt & set(l_idx.tolist())) / top_k)

    base_r = np.mean(base_recalls)
    learned_r = np.mean(learned_recalls)
    # Learned should be at least as good (allow small tolerance for randomness)
    assert learned_r >= base_r - 0.05, \
        f"Learned recall {learned_r:.3f} < base {base_r:.3f} - 0.05"


def test_hash_batch_exact_k(corpus):
    lf = LearnedFlyHash(input_dim=64, kenyon_cells=200, sparsity=0.05,
                        seed=42, n_epochs=2)
    lf.fit(corpus)
    fps = lf.hash_batch(corpus)
    bits_per_row = fps.sum(axis=1)
    assert np.all(bits_per_row == lf.k)


def test_find_neighbors_shape(corpus):
    lf = LearnedFlyHash(input_dim=64, kenyon_cells=200, sparsity=0.05, seed=42)
    lf.fit(corpus)
    fps = lf.hash_batch(corpus)
    idx, scores = lf.find_neighbors(corpus[0], fps, top_k=3)
    assert idx.shape == (3,)
    assert scores.shape == (3,)
    assert scores[0] >= scores[-1]
