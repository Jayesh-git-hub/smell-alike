"""
01_algorithm_walkthrough.py
============================
Phase 0 deliverable: understand the Fly LSH algorithm step by step.
Run this as a plain Python script to see the algorithm in action.

Reference: Dasgupta, Stevens & Navlakha (2017) Science 358:793
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from smell_alike.fly_hash import FlyHash

print("=" * 60)
print("  SMELL-ALIKE: Fly LSH Algorithm Walkthrough")
print("=" * 60)

# ── Setup ─────────────────────────────────────────────────────────────────────
INPUT_DIM    = 128   # stands in for "50 projection neurons" in the paper
KENYON_CELLS = 2000  # expansion layer (actual fly has ~2000)
SPARSITY     = 0.05  # top 5% survive WTA inhibition

hasher = FlyHash(input_dim=INPUT_DIM, kenyon_cells=KENYON_CELLS, sparsity=SPARSITY)
print(f"\nHasher: {hasher}")
print(f"Projection matrix W shape: {hasher.W.shape}")
nonzero_frac = np.count_nonzero(hasher.W) / hasher.W.size
print(f"Non-zero weights: {nonzero_frac:.1%}  (sparse connectivity, like real fly)")

# ── Step 1: Create three test vectors ─────────────────────────────────────────
rng = np.random.default_rng(7)

x_a = rng.standard_normal(INPUT_DIM)
x_a /= np.linalg.norm(x_a)          # L2-normalise

# Near-duplicate of x_a (small noise)
x_b = x_a + rng.standard_normal(INPUT_DIM) * 0.1
x_b /= np.linalg.norm(x_b)

# Completely unrelated vector
x_c = rng.standard_normal(INPUT_DIM)
x_c /= np.linalg.norm(x_c)

cos_ab = float(x_a @ x_b)
cos_ac = float(x_a @ x_c)
print(f"\nCosine similarity  x_a vs x_b (near-dup): {cos_ab:.4f}")
print(f"Cosine similarity  x_a vs x_c (random)  : {cos_ac:.4f}")

# ── Step 2: Hash all three ────────────────────────────────────────────────────
fp_a = hasher.hash(x_a)
fp_b = hasher.hash(x_b)
fp_c = hasher.hash(x_c)

print("\n--- FINGERPRINTS ---")
print(f"fp_a  bits set: {fp_a.sum()}  / {KENYON_CELLS}  ({fp_a.sum()/KENYON_CELLS:.1%})")
print(f"fp_b  bits set: {fp_b.sum()}  / {KENYON_CELLS}  ({fp_b.sum()/KENYON_CELLS:.1%})")
print(f"fp_c  bits set: {fp_c.sum()}  / {KENYON_CELLS}  ({fp_c.sum()/KENYON_CELLS:.1%})")

jac_ab = hasher.similarity(fp_a, fp_b)
jac_ac = hasher.similarity(fp_a, fp_c)
print(f"\n--- JACCARD SIMILARITY (fingerprint overlap) ---")
print(f"  x_a vs x_b (near-dup): {jac_ab:.4f}  <-- should be HIGH")
print(f"  x_a vs x_c (random)  : {jac_ac:.4f}  <-- should be LOW")

assert jac_ab > jac_ac, "FAIL: near-dup should score higher than random pair"
print("\n[PASS] Similar inputs have more overlapping fingerprints.")

# ── Step 3: Scale test ────────────────────────────────────────────────────────
print("\n--- SCALE TEST (1000-item corpus) ---")
corpus = rng.standard_normal((1000, INPUT_DIM))
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)

import time
t0 = time.perf_counter()
corpus_fps = hasher.hash_batch(corpus)
t1 = time.perf_counter()
print(f"Hashed 1000 vectors in {(t1-t0)*1000:.1f} ms")
print(f"Fingerprint matrix shape: {corpus_fps.shape}")

# Inject a known near-duplicate and verify it comes back top
query = corpus[42] + rng.standard_normal(INPUT_DIM) * 0.05
query /= np.linalg.norm(query)
idx, scores = hasher.find_neighbors(query, corpus_fps, top_k=5)
print(f"\nQuery is near-dup of corpus[42]")
print(f"Top-5 neighbor indices: {idx}")
print(f"Top-5 neighbor scores : {scores.round(4)}")
print(f"[{'PASS' if 42 in idx else 'NOTE: not in top-5, try reducing noise'}]  corpus[42] found: {42 in idx}")

report = hasher.sparsity_report(corpus_fps)
print(f"\nSparsity report: {report}")
print("\n" + "=" * 60)
print("  Phase 0 complete. Algorithm verified.")
print("  Run pytest next: pytest smell_alike/tests/ -v")
print("=" * 60)
