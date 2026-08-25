"""
datasets.py  --  Real-dataset loaders for honest benchmarking
=============================================================
The built-in 60-note sample corpus is fine for demos, but the project spec
calls for benchmarking on REAL data. This module provides loaders that
return (texts, labels) with natural topical clusters so Recall@K is
meaningful: the true top-K neighbours of a document are usually documents
from the same newsgroup/topic.

Datasets:
  - fetch_20newsgroups  -- classic text corpus, downloads ~14 MB on first
                           use and caches it in scikit-learn's data dir.

Usage:
    from smell_alike.datasets import load_20newsgroups
    texts, labels = load_20newsgroups(n_per_group=60)
"""
import numpy as np

# ── 20 Newsgroups ─────────────────────────────────────────────────────────────

DEFAULT_GROUPS = [
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.mideast",
    "comp.graphics",
    "sci.med",
]


def load_20newsgroups(n_per_group=60, groups=None, remove=("headers", "footers", "quotes"), seed=0):
    """
    Load a balanced subset of the 20 Newsgroups corpus.

    Headers/footers/quotes are stripped by default so documents are not
    trivially matched by thread headers — matching content, not metadata.

    Parameters
    ----------
    n_per_group  : int    -- documents per newsgroup
    groups       : list[str] | None -- which newsgroups (default: 5 diverse ones)
    remove       : tuple  -- passed to sklearn's fetch_20newsgroups
    seed         : int    -- RNG seed for the subsample (reproducible)

    Returns
    -------
    texts  : list[str]  -- document bodies (empty docs dropped)
    labels : list[str]  -- newsgroup name per document
    """
    from sklearn.datasets import fetch_20newsgroups

    chosen = groups or DEFAULT_GROUPS
    rng = np.random.default_rng(seed)

    texts, labels = [], []
    for g in chosen:
        docs = fetch_20newsgroups(
            subset="train", categories=[g], remove=remove, shuffle=True, random_state=seed
        ).data
        docs = [d.strip() for d in docs if len(d.strip()) > 40]  # drop empties
        take = min(n_per_group, len(docs))
        idx = rng.choice(len(docs), size=take, replace=False)
        texts.extend(docs[i] for i in idx)
        labels.extend([g] * take)

    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]
    return texts, labels


def load_dataset(name, **kwargs):
    """
    Loader registry for the benchmark CLI.

    name: "sample" -> built-in 60-note corpus (labels = category prefixes)
          "20ng"   -> 20 Newsgroups subset
    """
    name = name.lower()
    if name in ("sample", "sample60"):
        from smell_alike.embeddings import load_sample_corpus

        texts = load_sample_corpus()
        return texts, ["sample"] * len(texts)
    if name in ("20ng", "20newsgroups", "newsgroups"):
        return load_20newsgroups(**kwargs)
    raise ValueError(
        f"Unknown dataset {name!r}. Available: 'sample', '20ng'"
    )
