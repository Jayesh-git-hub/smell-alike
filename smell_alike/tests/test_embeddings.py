"""
test_embeddings.py
==================
Tests for TextEmbedder and ImageEmbedder (Phase 3 front-ends).

Heavy tests that download/load real models are skipped automatically when
the model is not cached locally, so the suite stays fast and offline-safe.
Set SMELL_ALIKE_TEST_MODELS=1 to force them to run (and download).

Run with:  pytest smell_alike/tests/test_embeddings.py -v
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from smell_alike.embeddings import (
    TextEmbedder,
    ImageEmbedder,
    load_synthetic_image_corpus,
    load_image_folder,
)

FORCE_MODELS = os.environ.get("SMELL_ALIKE_TEST_MODELS") == "1"


def _model_cached(model_name):
    """Best-effort check whether a HF model is already in the local cache."""
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub_dir = hf_cache / "hub"
    if not hub_dir.exists():
        return False
    return any(model_name.replace("/", "--") in p.name for p in hub_dir.iterdir())


def _skip_if_model_not_cached(model_name):
    if not FORCE_MODELS and not _model_cached(model_name):
        pytest.skip(f"model {model_name!r} not cached locally; "
                    f"set SMELL_ALIKE_TEST_MODELS=1 to download")


# ── TextEmbedder ──────────────────────────────────────────────────────────────

def test_text_embedder_repr():
    emb = TextEmbedder()
    assert "TextEmbedder" in repr(emb)
    assert "all-MiniLM-L6-v2" in repr(emb)


def test_text_embedder_encode():
    _skip_if_model_not_cached("all-MiniLM-L6-v2")
    emb = TextEmbedder()
    vecs = emb.encode(["the fly brain hashes smells", "kenyon cells fire sparsely"])
    assert vecs.shape == (2, emb.dim)
    assert vecs.dtype == np.float32
    # L2-normalised: every row norm ~1
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)


def test_text_embedder_paraphrases_closer_than_random():
    _skip_if_model_not_cached("all-MiniLM-L6-v2")
    emb = TextEmbedder()
    v = emb.encode([
        "How do neural circuits implement algorithms?",
        "What algorithms do neural circuits implement?",
        "The price of tea in China is quite variable.",
    ])
    cos = v @ v.T
    assert cos[0, 1] > cos[0, 2]   # paraphrase beats unrelated


# ── ImageEmbedder ─────────────────────────────────────────────────────────────

def test_image_embedder_repr():
    emb = ImageEmbedder()
    assert "ImageEmbedder" in repr(emb)
    assert "clip-ViT-B-32" in repr(emb)


def test_image_embedder_encode_synthetic():
    _skip_if_model_not_cached("clip-ViT-B-32")
    emb = ImageEmbedder()
    images, labels = load_synthetic_image_corpus(n=8, size=64)
    vecs = emb.encode(images)
    assert vecs.shape == (8, emb.dim)
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)


def test_image_embedder_accepts_numpy_arrays():
    _skip_if_model_not_cached("clip-ViT-B-32")
    emb = ImageEmbedder()
    rng = np.random.default_rng(0)
    arrs = [rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
    vecs = emb.encode(arrs)
    assert vecs.shape == (3, emb.dim)


# ── Image corpus helpers ──────────────────────────────────────────────────────

def test_synthetic_corpus_groups():
    images, labels = load_synthetic_image_corpus(n=12, seed=3)
    assert len(images) == len(labels) == 12
    # 4 kinds cycling -> exactly 3 images per kind
    kinds = {lbl.rsplit("-", 1)[0] for lbl in labels}
    assert kinds == {"h-stripes", "v-stripes", "diag", "block"}
    for kind in kinds:
        assert sum(1 for l in labels if l.rsplit("-", 1)[0] == kind) == 3


def test_load_image_folder_roundtrip(tmp_path):
    from PIL import Image

    for i, color in enumerate([(255, 0, 0), (0, 255, 0)]):
        Image.new("RGB", (10, 10), color).save(tmp_path / f"img_{i}.png")
    (tmp_path / "ignore.txt").write_text("not an image")

    paths = load_image_folder(str(tmp_path))
    assert len(paths) == 2
    assert all(p.endswith(".png") for p in paths)


def test_load_image_folder_empty_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_folder(str(tmp_path))
