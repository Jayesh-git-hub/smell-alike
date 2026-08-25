"""
connectome.py  --  Real & synthetic connectome projection matrices
==================================================================
Provides projection matrices W based on actual fly connectome data
or statistical approximations thereof.

The real Drosophila olfactory circuit has:
  - ~50 PN types (projection neurons)
  - ~2000 KCs  (Kenyon cells)
  - Each KC receives input from ~6 random PNs (6% connectivity)

Data sources:
  - FlyWire  (Dorkenwald et al. 2023, Nature)
  - hemibrain (Scheffer et al. 2020, eLife)

This module can:
  1. Generate a synthetic W that matches the real statistics
  2. Load a real adjacency matrix from a CSV if provided
  3. Adapt a PN-level connectivity matrix to arbitrary input_dim
"""
import numpy as np
from pathlib import Path


# ── Known statistics from the literature ──────────────────────────────────────
REAL_STATS = {
    "n_pn_types":       50,
    "n_kenyon_cells":   2000,
    "mean_pn_per_kc":   6,       # each KC samples ~6 PNs
    "connectivity":     0.06,    # 6/50 ≈ 0.12, but with weight variation ~0.06
    "kc_sparsity":      0.05,    # ~5% of KCs active for any odour
}


def connectome_projection(input_dim, kenyon_cells=2000, seed=42,
                          connectivity=None, csv_path=None):
    """
    Build a projection matrix W that mirrors real fly wiring.

    Parameters
    ----------
    input_dim     : int          -- embedding dimensionality (e.g. 384)
    kenyon_cells  : int          -- number of KCs (default 2000)
    seed          : int          -- RNG seed
    connectivity  : float | None -- override wiring density (default: from stats)
    csv_path      : str | None   -- path to a real adjacency CSV (PN x KC)

    Returns
    -------
    W : (kenyon_cells, input_dim) float64 sparse projection matrix
    """
    if csv_path is not None:
        return _load_real_connectome(csv_path, input_dim, kenyon_cells, seed)

    conn = connectivity or REAL_STATS["connectivity"]
    return _synthetic_connectome(input_dim, kenyon_cells, conn, seed)


def _synthetic_connectome(input_dim, kenyon_cells, connectivity, seed):
    """
    Generate W matching real statistics:
    - Each KC connects to ~connectivity fraction of inputs
    - Non-zero weights drawn from log-normal (mimicking synaptic strength variation)
    - Rows L2-normalised so KCs have comparable activation magnitudes
    """
    rng = np.random.default_rng(seed)

    # Binary mask: each KC connects to ~connectivity*input_dim inputs
    mask = rng.random((kenyon_cells, input_dim)) < connectivity

    # Synaptic weights: log-normal models the heavy-tailed weight distribution
    # seen in real connectomes (few strong synapses, many weak ones)
    weights = rng.lognormal(mean=0.0, sigma=0.5, size=(kenyon_cells, input_dim))
    W = weights * mask

    # L2-normalise each KC row (so activation magnitude depends on input, not wiring)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid div by zero for disconnected KCs
    W = W / norms

    return W


def _load_real_connectome(csv_path, input_dim, kenyon_cells, seed):
    """
    Load a real PN→KC adjacency matrix from CSV and adapt it to input_dim.

    Expected CSV format: rows = PNs, columns = KCs, values = synapse counts.
    The matrix is transposed to (KC, PN) then projected to (KC, input_dim)
    via a random linear map if PN count != input_dim.
    """
    import pandas as pd

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Connectome CSV not found: {path}")

    df = pd.read_csv(path, index_col=0)
    adj = df.values.astype(np.float64)  # (n_pn, n_kc)

    # Transpose to (KC, PN)
    W_pn = adj.T  # (n_kc_real, n_pn_real)
    n_kc_real, n_pn_real = W_pn.shape

    # Resize KC dimension if needed (subsample or tile)
    if n_kc_real != kenyon_cells:
        rng = np.random.default_rng(seed)
        if n_kc_real > kenyon_cells:
            idx = rng.choice(n_kc_real, size=kenyon_cells, replace=False)
            W_pn = W_pn[idx]
        else:
            reps = kenyon_cells // n_kc_real + 1
            W_pn = np.tile(W_pn, (reps, 1))[:kenyon_cells]

    # Project from PN space to input_dim via random linear map
    if n_pn_real != input_dim:
        rng = np.random.default_rng(seed + 1)
        P = rng.standard_normal((n_pn_real, input_dim)) / np.sqrt(n_pn_real)
        W = W_pn @ P
    else:
        W = W_pn

    # L2-normalise rows
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    W = W / norms

    return W


class ConnectomeFlyHash:
    """
    FlyHash variant using a connectome-derived projection matrix.

    Drop-in replacement: same hash/hash_batch/find_neighbors interface.
    """

    def __init__(self, input_dim, kenyon_cells=2000, sparsity=0.05,
                 seed=42, csv_path=None, connectivity=None):
        self.input_dim = input_dim
        self.kenyon_cells = kenyon_cells
        self.sparsity = sparsity
        self.seed = seed
        self.k = max(1, int(kenyon_cells * sparsity))

        self.W = connectome_projection(
            input_dim, kenyon_cells, seed,
            connectivity=connectivity, csv_path=csv_path,
        )

    def hash(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        act = self.W @ x
        top_idx = np.argpartition(act, -self.k)[-self.k:]
        fp = np.zeros(self.kenyon_cells, dtype=np.uint8)
        fp[top_idx] = 1
        return fp

    def hash_batch(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            return self.hash(X)[np.newaxis, :]
        ACT = X @ self.W.T
        top_idx = np.argpartition(ACT, -self.k, axis=1)[:, -self.k:]
        FP = np.zeros_like(ACT, dtype=np.uint8)
        rows = np.arange(ACT.shape[0])[:, np.newaxis]
        FP[rows, top_idx] = 1
        return FP

    def find_neighbors(self, query, corpus_fps, top_k=5):
        q = self.hash(query)
        inter = (corpus_fps & q).sum(axis=1).astype(np.float64)
        union = (corpus_fps | q).sum(axis=1).astype(np.float64)
        scores = np.where(union > 0, inter / union, 0.0)
        idx = np.argsort(-scores)[:top_k]
        return idx, scores[idx]

    def __repr__(self):
        return (f"ConnectomeFlyHash(input_dim={self.input_dim}, "
                f"kenyon_cells={self.kenyon_cells}, "
                f"sparsity={self.sparsity}, k={self.k})")
