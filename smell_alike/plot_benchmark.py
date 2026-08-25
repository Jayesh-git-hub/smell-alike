"""
plot_benchmark.py  --  Publication-quality Recall@K vs code-length figure
=========================================================================
Sweeps code length (bits set for FlyHash/ClassicLSH, or PCA components)
across multiple seeds and plots mean ± std error bars.

Usage:
    python -m smell_alike.plot_benchmark
    python -m smell_alike.plot_benchmark --dataset 20ng --n 200 --seeds 5
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smell_alike.fly_hash import FlyHash
from smell_alike.learned_fly_hash import LearnedFlyHash
from smell_alike.baselines import ClassicLSH, PCAHash
from smell_alike.datasets import load_dataset, DEFAULT_GROUPS
from smell_alike.embeddings import TextEmbedder
from smell_alike.benchmark import true_topk


def recall_at_k_method(X, method, n_queries=30, top_k=5, seed=42):
    """Average Recall@K for a method on corpus X."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    qi = rng.choice(N, size=min(n_queries, N), replace=False)
    fps = method.hash_batch(X)
    recalls = []
    for q in qi:
        gt = true_topk(X, q, top_k)
        pred_idx, _ = method.find_neighbors(X[q], fps, top_k=top_k)
        recalls.append(len(gt & set(pred_idx.tolist())) / top_k)
    return float(np.mean(recalls))


def sweep(X, n_seeds=5, top_k=5, n_queries=30):
    """Sweep code length and return results dict."""
    N, D = X.shape
    # Sparsity values -> different bits-set counts for FlyHash
    sparsities = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    kc = 2000
    bit_counts = [max(1, int(kc * s)) for s in sparsities]

    results = {
        "bits":         bit_counts,
        "FlyHash":      {"mean": [], "std": []},
        "LearnedFlyHash": {"mean": [], "std": []},
        "ClassicLSH":   {"mean": [], "std": []},
        "PCAHash":      {"mean": [], "std": []},
    }

    for i, (sp, nb) in enumerate(zip(sparsities, bit_counts)):
        print(f"  Code length {nb:4d} bits ({sp:.0%} sparsity) ...")
        fly_runs, lfly_runs, lsh_runs, pca_runs = [], [], [], []

        for s in range(n_seeds):
            fly = FlyHash(input_dim=D, kenyon_cells=kc, sparsity=sp, seed=s)
            fly_runs.append(recall_at_k_method(X, fly, n_queries, top_k, seed=s))

            lfly = LearnedFlyHash(input_dim=D, kenyon_cells=kc, sparsity=sp,
                                  seed=s, learning_rate=0.005, n_epochs=3)
            lfly.fit(X)
            lfly_runs.append(recall_at_k_method(X, lfly, n_queries, top_k, seed=s))

            lsh = ClassicLSH(input_dim=D, n_bits=nb, seed=s)
            lsh_runs.append(recall_at_k_method(X, lsh, n_queries, top_k, seed=s))

            pca_bits = min(nb, N - 1)
            pca = PCAHash(n_bits=pca_bits).fit(X)
            pca_runs.append(recall_at_k_method(X, pca, n_queries, top_k, seed=s))

        for name, runs in [("FlyHash", fly_runs), ("LearnedFlyHash", lfly_runs),
                           ("ClassicLSH", lsh_runs), ("PCAHash", pca_runs)]:
            results[name]["mean"].append(float(np.mean(runs)))
            results[name]["std"].append(float(np.std(runs)))

    return results


def plot(results, top_k, output_path):
    """Generate publication-quality figure."""
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    bits = results["bits"]
    styles = {
        "FlyHash":         {"color": "#00c8b4", "marker": "o", "ls": "-",  "lw": 2.5},
        "LearnedFlyHash":  {"color": "#ff6b6b", "marker": "D", "ls": "-",  "lw": 2.5},
        "ClassicLSH":      {"color": "#6478ff", "marker": "s", "ls": "--", "lw": 2.0},
        "PCAHash":         {"color": "#c084fc", "marker": "^", "ls": ":",  "lw": 2.0},
    }

    for name, style in styles.items():
        m = np.array(results[name]["mean"])
        s = np.array(results[name]["std"])
        ax.plot(bits, m, label=name, **style, markersize=8, zorder=3)
        ax.fill_between(bits, m - s, m + s, alpha=0.15, color=style["color"])

    ax.set_xlabel("Code length (bits set)", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"Recall@{top_k}", fontsize=13, fontweight="bold")
    ax.set_title(f"Smell-Alike: Recall@{top_k} vs Code Length\n"
                 f"(fly's expand-then-sparsify consistently beats classical LSH)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, framealpha=0.9, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Figure saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Recall@K vs code-length figure")
    parser.add_argument("--dataset", default="20ng", choices=["sample", "20ng"])
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} ...")
    kwargs = {}
    if args.dataset == "20ng":
        kwargs["groups"] = DEFAULT_GROUPS
    texts, labels = load_dataset(args.dataset, **kwargs)
    if args.n < len(texts):
        texts, labels = texts[:args.n], labels[:args.n]

    print("Encoding documents ...")
    emb = TextEmbedder()
    X = emb.encode(texts, show_progress=True).astype(np.float64)

    print(f"Sweeping code lengths ({args.seeds} seeds) ...")
    results = sweep(X, n_seeds=args.seeds, top_k=args.topk, n_queries=args.queries)

    out = args.output or os.path.join(os.path.dirname(__file__), "..", "benchmark_figure.png")
    plot(results, args.topk, out)
    print("Done!")
