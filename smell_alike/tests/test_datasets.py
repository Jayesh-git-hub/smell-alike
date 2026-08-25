"""
test_datasets.py
================
Tests for dataset loaders. Requires network + sklearn data on first use;
skipped automatically when the 20 Newsgroups cache is absent.
Set SMELL_ALIKE_TEST_MODELS=1 to force them to run (and download).
"""
import os
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest

from smell_alike.datasets import load_20newsgroups, load_dataset

FORCE_MODELS = os.environ.get("SMELL_ALIKE_TEST_MODELS") == "1"


def _20ng_cached():
    sklearn_data = os.path.join(os.path.expanduser("~"), "scikit_learn_data")
    return os.path.isdir(sklearn_data) and any(
        "20news" in f for f in os.listdir(sklearn_data)
    )


@pytest.mark.skipif(
    "not FORCE_MODELS and not _20ng_cached()",
    reason="20 Newsgroups not cached; set SMELL_ALIKE_TEST_MODELS=1 to download",
)
class Test20Newsgroups:
    def test_load_subset(self):
        texts, labels = load_20newsgroups(n_per_group=10)
        assert len(texts) == len(labels) == 50   # 5 default groups x 10
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)

    def test_balanced_labels(self):
        texts, labels = load_20newsgroups(n_per_group=10)
        from collections import Counter
        counts = Counter(labels)
        assert set(counts.values()) == {10}

    def test_load_dataset_registry(self):
        texts, labels = load_dataset("20ng", n_per_group=5)
        assert len(texts) == 25


def test_load_dataset_sample():
    texts, labels = load_dataset("sample")
    assert len(texts) == 60
    assert set(labels) == {"sample"}


def test_load_dataset_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("nope")
