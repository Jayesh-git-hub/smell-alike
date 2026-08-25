"""
benchmark.py  --  Phase 4: Head-to-head comparison
====================================================
Metrics:
  Recall@K     -- fraction of true top-K (cosine) neighbours found
  Purity@K     -- fraction of top-K results sharing the query's label
                  (only shown when the dataset has real labels)
  QPS          -- query throughput (queries per second)
  Memory KB    -- total fingerprint storage

Fairness -- matched code length:
  Every method stores a binary code; comparing a 2000-bit code against a
  100-bit code is not a fair fight. Two match modes:

    --match bits     baselines use n_bits = FlyHash bits-set (default).
                     Same number of 1s per code -> comparable Jaccard /
                     Hamming behaviour. Tests the *expansion hypothesis*
                     itself. FlyHash pays more memory (2000 vs 100 bytes);
                     that cost is reported honestly in the memory column.
    --match storage  baselines use n_bits = FlyHash code length
                     (kenyon_cells). Equal storage per fingerprint.

Usage:
    python -m smell_alike.benchmark
    python -m smell_alike.benchmark --dataset 20ng --n 100 --topk 10
    python -m smell_alike.benchmark --dataset 20ng --match storage --bits 2000
"""
import numpy as np
import time
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smell_alike.fly_hash import FlyHash
from smell_alike.learned_fly_hash import LearnedFlyHash
from smell_alike.connectome import ConnectomeFlyHash
from smell_alike.baselines import ClassicLSH, PCAHash
from smell_alike.embeddings import TextEmbedder
from smell_alike.datasets import load_dataset, DEFAULT_GROUPS


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def true_topk(X, query_idx, k):
    """Brute-force cosine nearest neighbours (ground truth)."""
    q = X[query_idx]
    sims = X @ q
    sims[query_idx] = -1   # exclude self
    return set(np.argsort(-sims)[:k])


def recall_at_k(X, corpus_fps, method, query_indices, k):
    """Average Recall@k over a set of query indices."""
    recalls = []
    for qi in query_indices:
        gt = true_topk(X, qi, k)
        pred_idx, _ = method.find_neighbors(X[qi], corpus_fps, top_k=k)
        pred = set(pred_idx.tolist())
        recalls.append(len(gt & pred) / k)
    return float(np.mean(recalls))


def purity_at_k(X, corpus_fps, method, query_indices, labels, k):
    """Average fraction of top-K results sharing the query's label."""
    purities = []
    for qi in query_indices:
        pred_idx, _ = method.find_neighbors(X[qi], corpus_fps, top_k=k)
        purities.append(np.mean([labels[j] == labels[qi] for j in pred_idx]))
    return float(np.mean(purities))


def throughput_qps(X, corpus_fps, method, n_queries=200):
    """Queries per second (includes hashing the query, as in real use)."""
    rng = np.random.default_rng(0)
    qs = X[rng.choice(len(X), size=n_queries, replace=False)]
    t0 = time.perf_counter()
    for q in qs:
        method.find_neighbors(q, corpus_fps)
    elapsed = time.perf_counter() - t0
    return n_queries / elapsed


def memory_kb(corpus_fps):
    """Total fingerprint storage in KB."""
    return corpus_fps.nbytes / 1024


# -----------------------------------------------------------------------------
# Main benchmark
# -----------------------------------------------------------------------------

def run_benchmark(X, labels=None, top_k=5, n_queries=50, match="bits",
                  baseline_bits=100, kenyon_cells=2000, sparsity=0.05):
    """
    Run all methods on corpus X and return a results dict.

    Parameters
    ----------
    X             : (N, D) L2-normalised float64 corpus
    labels        : list[str] of length N (optional; enables Purity@K)
    top_k         : K for Recall@K
    n_queries     : number of random queries to evaluate
    match         : "bits"    -> baselines get FlyHash's bits-set as code length
                    "storage" -> baselines get `baseline_bits` as code length
    baseline_bits : code length for baselines in "storage" mode
    """
    N, D = X.shape
    rng = np.random.default_rng(42)
    query_indices = rng.choice(N, size=min(n_queries, N), replace=False).tolist()

    # -- Initialise methods ----------------------------------------------------
    fly = FlyHash(input_dim=D, kenyon_cells=kenyon_cells, sparsity=sparsity, seed=42)

    if match == "bits":
        n_bits = fly.k                      # same # of set bits as FlyHash
        match_desc = f"{n_bits} bits (matched to FlyHash bits-set)"
    elif match == "storage":
        n_bits = baseline_bits
        match_desc = f"{n_bits} bits (matched to FlyHash storage)"
    else:
        raise ValueError(f"match must be 'bits' or 'storage', got {match!r}")

    lsh = ClassicLSH(input_dim=D, n_bits=n_bits, seed=0)
    pca_bits = min(n_bits, N - 1)
    pca = PCAHash(n_bits=pca_bits).fit(X)

    # Learned variant (Hebbian)
    learned = LearnedFlyHash(input_dim=D, kenyon_cells=kenyon_cells,
                             sparsity=sparsity, seed=42,
                             learning_rate=0.005, n_epochs=3)
    learned.fit(X, verbose=False)

    # Connectome variant (bio-realistic wiring)
    conn = ConnectomeFlyHash(input_dim=D, kenyon_cells=kenyon_cells,
                             sparsity=sparsity, seed=42)

    methods = {
        "FlyHash (random W)":      fly,
        "LearnedFly (Hebbian)":    learned,
        "Connectome (bio-wiring)": conn,
        "ClassicLSH (baseline)":   lsh,
        "PCAHash (data-driven)":   pca,
    }

    has_labels = labels is not None and len(set(labels)) > 1
    results = {}

    print(f"\n{'='*72}")
    print(f"  Smell-Alike Benchmark")
    print(f"  Corpus: {N} items  |  Dim: {D}  |  Recall@{top_k}  |  {len(query_indices)} queries")
    print(f"  FlyHash: {kenyon_cells} KCs x {sparsity:.0%} = {fly.k} bits set"
          f"  |  Baselines: {match_desc}")
    print(f"{'='*72}")
    purity_col = f"{'Purity@'+str(top_k):<14}" if has_labels else ""
    print(f"{'Method':<26} {'Recall@'+str(top_k):<14} {purity_col} {'QPS':<12} {'Memory KB':<12}")
    print(f"{'-'*72}")

    for name, method in methods.items():
        fps = method.hash_batch(X)

        rec = recall_at_k(X, fps, method, query_indices, top_k)
        qps = throughput_qps(X, fps, method, n_queries=min(n_queries, N))
        mem = memory_kb(fps)
        pur = purity_at_k(X, fps, method, query_indices, labels, top_k) if has_labels else None

        results[name] = {"recall": rec, "qps": qps, "memory_kb": mem, "purity": pur}
        pur_str = f"{pur:<14.3f}" if pur is not None else ""
        print(f"{name:<26} {rec:<14.3f} {pur_str} {qps:<12.0f} {mem:<12.1f}")

    print(f"{'='*72}\n")
    return results


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smell-Alike benchmark")
    parser.add_argument("--dataset", type=str, default="sample",
                        choices=["sample", "20ng"],
                        help="corpus to benchmark on (default: built-in sample)")
    parser.add_argument("--n",       type=int, default=60,
                        help="max corpus size (default: 60)")
    parser.add_argument("--groups",  type=int, default=5,
                        help="20ng: number of newsgroups (default: 5)")
    parser.add_argument("--per-group", type=int, default=60,
                        help="20ng: docs per newsgroup (default: 60)")
    parser.add_argument("--topk",    type=int, default=5,  help="K for Recall@K")
    parser.add_argument("--queries", type=int, default=30, help="number of query vectors")
    parser.add_argument("--match",   type=str, default="bits",
                        choices=["bits", "storage"],
                        help="code-length matching mode (default: bits)")
    parser.add_argument("--bits",    type=int, default=2000,
                        help="baseline code length in --match storage mode")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} ...")
    kwargs = {"n_per_group": args.per_group}
    if args.dataset == "20ng":
        kwargs["groups"] = DEFAULT_GROUPS[:args.groups]
    texts, labels = load_dataset(args.dataset, **kwargs)
    if args.n < len(texts):
        texts, labels = texts[:args.n], labels[:args.n]

    print("Loading embeddings...")
    emb = TextEmbedder()
    print(f"Encoding {len(texts)} documents...")
    X = emb.encode(texts, show_progress=True).astype(np.float64)

    run_benchmark(X, labels=labels, top_k=args.topk,
                  n_queries=args.queries, match=args.match,
                  baseline_bits=args.bits)
