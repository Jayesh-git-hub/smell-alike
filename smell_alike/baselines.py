"""
baselines.py  --  Two baseline hashers to benchmark FlyHash against
====================================================================
1. ClassicLSH  -- standard random-projection LSH (dense, compressed)
2. PCAHash     -- data-driven projection using PCA components
"""
import numpy as np

try:
    from sklearn.decomposition import PCA
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


class ClassicLSH:
    """
    Standard random-projection LSH.

    Unlike FlyHash, this COMPRESSES: D -> n_bits (n_bits << D usually).
    Each bit is the sign of a dot product with a random Gaussian vector.
    This is the Charikar (2002) SimHash / random-projection LSH.

    Parameters
    ----------
    input_dim : dimensionality of L2-normalised input vectors
    n_bits    : number of hash bits (code length)
    seed      : random seed
    """

    def __init__(self, input_dim, n_bits=100, seed=0):
        self.input_dim = input_dim
        self.n_bits = n_bits
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.R = rng.standard_normal((n_bits, input_dim))  # (bits, D)

    def hash(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        return (self.R @ x > 0).astype(np.uint8)

    def hash_batch(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X @ self.R.T > 0).astype(np.uint8)

    def find_neighbors(self, query, corpus_fps, top_k=5):
        q = self.hash(query)
        # Hamming-based similarity: (bits - hamming) / bits
        hamming = (corpus_fps ^ q).sum(axis=1)
        scores = 1.0 - hamming / self.n_bits
        idx = np.argsort(-scores)[:top_k]
        return idx, scores[idx]

    def __repr__(self):
        return f"ClassicLSH(input_dim={self.input_dim}, n_bits={self.n_bits})"


class PCAHash:
    """
    Data-driven hashing via PCA projection.

    Fits PCA on corpus, then binarises the principal components.
    Must call .fit(X) before .hash().

    Parameters
    ----------
    n_bits : number of PCA components to keep as bits
    """

    def __init__(self, n_bits=100):
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for PCAHash. pip install scikit-learn")
        self.n_bits = n_bits
        self.pca = None
        self._thresholds = None

    def fit(self, X):
        """Fit PCA on corpus X: (N, D)."""
        X = np.asarray(X, dtype=np.float64)
        self.pca = PCA(n_components=self.n_bits)
        projected = self.pca.fit_transform(X)
        self._thresholds = np.median(projected, axis=0)  # per-component median threshold
        return self

    def hash(self, x):
        if self.pca is None:
            raise RuntimeError("Call .fit(X) before .hash()")
        x = np.asarray(x, dtype=np.float64).ravel()
        proj = self.pca.transform(x[np.newaxis, :])[0]
        return (proj > self._thresholds).astype(np.uint8)

    def hash_batch(self, X):
        if self.pca is None:
            raise RuntimeError("Call .fit(X) before .hash_batch()")
        X = np.asarray(X, dtype=np.float64)
        proj = self.pca.transform(X)
        return (proj > self._thresholds).astype(np.uint8)

    def find_neighbors(self, query, corpus_fps, top_k=5):
        q = self.hash(query)
        hamming = (corpus_fps ^ q).sum(axis=1)
        scores = 1.0 - hamming / self.n_bits
        idx = np.argsort(-scores)[:top_k]
        return idx, scores[idx]

    def __repr__(self):
        return f"PCAHash(n_bits={self.n_bits}, fitted={self.pca is not None})"
