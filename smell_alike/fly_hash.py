"""
fly_hash.py  --  Drosophila-inspired Locality-Sensitive Hashing
================================================================
Based on: Dasgupta, Stevens & Navlakha (2017) Science 358:793-796
https://doi.org/10.1126/science.aam8084

The fly olfactory circuit runs 3 steps:
  1. EXPAND   -- sparse random projection: D -> 2000 Kenyon cells
  2. FIRE     -- activations = W @ x
  3. SPARSIFY -- winner-take-all: keep top 5% of activations -> binary tag

Key insight: similar smells -> overlapping tags -> neighbours found fast.
"""
import numpy as np


class FlyHash:
    """
    Bio-inspired LSH based on the Drosophila olfactory circuit.

    Parameters
    ----------
    input_dim      : dimensionality of (L2-normalised) input vectors
    kenyon_cells   : expansion layer size (default 2000, as in the fly)
    sparsity       : fraction of KCs kept after WTA inhibition (default 0.05)
    pn_connectivity: fraction of inputs each KC wires to   (default 0.06)
    seed           : random seed for reproducibility
    """

    def __init__(self, input_dim, kenyon_cells=2000, sparsity=0.05,
                 pn_connectivity=0.06, seed=42):
        if not 0 < sparsity < 1:
            raise ValueError(f"sparsity must be in (0,1), got {sparsity}")
        if not 0 < pn_connectivity < 1:
            raise ValueError(f"pn_connectivity must be in (0,1), got {pn_connectivity}")

        self.input_dim = input_dim
        self.kenyon_cells = kenyon_cells
        self.sparsity = sparsity
        self.pn_connectivity = pn_connectivity
        self.seed = seed
        self.k = max(1, int(kenyon_cells * sparsity))  # bits kept after WTA

        rng = np.random.default_rng(seed)
        W = rng.standard_normal((kenyon_cells, input_dim))
        # Sparse connectivity: zero out (1 - pn_connectivity) of weights
        W[rng.random(W.shape) > pn_connectivity] = 0.0
        self.W = W  # shape: (KC, D)

    # ------------------------------------------------------------------ hash
    def hash(self, x):
        """Single vector -> binary fingerprint (uint8, shape kenyon_cells)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim}-D, got {x.shape[0]}")
        act = self.W @ x                                  # EXPAND + FIRE
        # Deterministic WTA: pick exactly self.k winners;
        # np.argpartition is stable on ties via index order.
        top_idx = np.argpartition(act, -self.k)[-self.k:]
        fp = np.zeros(self.kenyon_cells, dtype=np.uint8)
        fp[top_idx] = 1
        return fp                                          # SPARSIFY

    def hash_batch(self, X):
        """
        (N, D) matrix -> (N, KC) uint8 fingerprint matrix.

        Vectorised: one matmul + per-row WTA threshold. Produces results
        bit-identical to calling hash() row by row.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            return self.hash(X)[np.newaxis, :]
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim}-D rows, got {X.shape[1]}")
        ACT = X @ self.W.T                                        # EXPAND + FIRE
        # Deterministic WTA: exactly self.k bits per row
        top_idx = np.argpartition(ACT, -self.k, axis=1)[:, -self.k:]
        FP = np.zeros_like(ACT, dtype=np.uint8)
        rows = np.arange(ACT.shape[0])[:, np.newaxis]
        FP[rows, top_idx] = 1
        return FP                                                  # SPARSIFY

    # ------------------------------------------------------------ similarity
    def similarity(self, fp_a, fp_b):
        """Jaccard similarity in [0, 1]. Higher = more similar."""
        inter = int(np.count_nonzero(fp_a & fp_b))
        union = int(np.count_nonzero(fp_a | fp_b))
        return inter / union if union > 0 else 0.0

    # ----------------------------------------------------------- knn search
    def find_neighbors(self, query, corpus_fps, top_k=5):
        """
        Nearest-neighbour search over a pre-hashed corpus.

        Parameters
        ----------
        query       : (D,) L2-normalised query vector
        corpus_fps  : (N, KC) pre-computed fingerprint matrix
        top_k       : number of results to return

        Returns
        -------
        indices : (top_k,) int array
        scores  : (top_k,) float Jaccard similarities, descending
        """
        q = self.hash(query)
        inter = (corpus_fps & q).sum(axis=1).astype(np.float64)
        union = (corpus_fps | q).sum(axis=1).astype(np.float64)
        scores = np.where(union > 0, inter / union, 0.0)
        idx = np.argsort(-scores)[:top_k]
        return idx, scores[idx]

    # ------------------------------------------------------------ diagnostics
    def sparsity_report(self, fingerprints):
        """Check actual sparsity of a batch of fingerprints."""
        bc = fingerprints.sum(axis=1)
        return {
            "target_sparsity": self.sparsity,
            "target_k":        self.k,
            "mean_bits_set":   float(bc.mean()),
            "actual_sparsity": float(bc.mean() / self.kenyon_cells),
            "std_bits_set":    float(bc.std()),
        }

    def __repr__(self):
        return (f"FlyHash(input_dim={self.input_dim}, "
                f"kenyon_cells={self.kenyon_cells}, "
                f"sparsity={self.sparsity}, k={self.k})")
