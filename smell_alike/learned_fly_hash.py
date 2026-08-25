"""
learned_fly_hash.py  --  Hebbian-learned FlyHash (v2 headline feature)
=====================================================================
Instead of a fixed random projection W, this module learns W from data
using a bio-plausible Hebbian update rule.  The key insight from
Dasgupta et al. (2018, PNAS) and the FlyHash literature:

    W_new = W_old + eta * (h_sparse @ x.T)

where h_sparse is the post-WTA binary activation.  This strengthens
connections between co-firing Kenyon cells and the input features that
triggered them -- exactly what Hebbian plasticity does in real flies.

The learned version should beat random FlyHash, which already beats
ClassicLSH.  Story: untrained fly > baselines > trained fly > untrained fly.

Interface: drop-in replacement for FlyHash (same hash/hash_batch/find_neighbors).
"""
import numpy as np
from smell_alike.fly_hash import FlyHash


class LearnedFlyHash(FlyHash):
    """
    FlyHash with Hebbian-learned projection weights.

    After construction the projection matrix W is random (identical to FlyHash).
    Call .fit(X) to refine W with Hebbian learning on corpus X.

    Parameters
    ----------
    input_dim       : int   -- dimensionality of input vectors
    kenyon_cells    : int   -- expansion size (default 2000)
    sparsity        : float -- WTA survivor fraction (default 0.05)
    pn_connectivity : float -- initial sparse wiring density (default 0.06)
    seed            : int   -- random seed
    learning_rate   : float -- Hebbian step size (default 0.01)
    n_epochs        : int   -- training passes over data (default 5)
    """

    def __init__(self, input_dim, kenyon_cells=2000, sparsity=0.05,
                 pn_connectivity=0.06, seed=42,
                 learning_rate=0.01, n_epochs=5):
        super().__init__(input_dim=input_dim, kenyon_cells=kenyon_cells,
                         sparsity=sparsity, pn_connectivity=pn_connectivity,
                         seed=seed)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, X, verbose=False):
        """
        Refine projection matrix W using Hebbian learning on corpus X.

        Parameters
        ----------
        X       : (N, D) L2-normalised float64 corpus
        verbose : bool  -- print loss per epoch

        Returns
        -------
        self (for chaining)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim}-D, got {X.shape[1]}")

        N = X.shape[0]
        rng = np.random.default_rng(self.seed + 1000)  # separate from init seed

        for epoch in range(self.n_epochs):
            order = rng.permutation(N)
            for i in order:
                x = X[i]                                   # (D,)
                act = self.W @ x                           # (KC,)
                # WTA: exact k winners via argpartition
                top_idx = np.argpartition(act, -self.k)[-self.k:]
                h = np.zeros(self.kenyon_cells, dtype=np.float64)
                h[top_idx] = 1.0                           # (KC,)

                # Hebbian update: strengthen connections for active pairs
                # delta_W = eta * outer(h, x)  but only for active KCs
                self.W[top_idx] += self.learning_rate * np.outer(h[top_idx], x)

            if verbose:
                # Quick self-recall check
                fps = self.hash_batch(X)
                overlaps = []
                for j in range(min(50, N)):
                    idx, _ = self.find_neighbors(X[j], fps, top_k=5)
                    overlaps.append(j in idx)
                self_recall = np.mean(overlaps)
                print(f"  Epoch {epoch+1}/{self.n_epochs}  self-recall@5: {self_recall:.3f}")

        self._fitted = True
        return self

    @property
    def is_fitted(self):
        return self._fitted

    def __repr__(self):
        status = "fitted" if self._fitted else "unfitted"
        return (f"LearnedFlyHash(input_dim={self.input_dim}, "
                f"kenyon_cells={self.kenyon_cells}, "
                f"sparsity={self.sparsity}, k={self.k}, "
                f"lr={self.learning_rate}, epochs={self.n_epochs}, "
                f"{status})")
