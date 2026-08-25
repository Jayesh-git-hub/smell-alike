"""
api.py  --  FastAPI backend for Smell-Alike v2
===============================================
REST API wrapping FlyHash, LearnedFlyHash, ConnectomeFlyHash.

Run:  uvicorn api:app --reload --port 8000
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from smell_alike.fly_hash import FlyHash
from smell_alike.learned_fly_hash import LearnedFlyHash
from smell_alike.connectome import ConnectomeFlyHash
from smell_alike.embeddings import TextEmbedder, load_sample_corpus
from smell_alike.benchmark import run_benchmark

app = FastAPI(title="Smell-Alike API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
embedder = None
corpus = None
corpus_vecs = None
hashers = {}
corpus_fps = {}


def get_embedder():
    global embedder
    if embedder is None:
        embedder = TextEmbedder()
    return embedder


def get_corpus():
    global corpus, corpus_vecs
    if corpus is None:
        corpus = load_sample_corpus()
        corpus_vecs = get_embedder().encode(corpus).astype(np.float64)
    return corpus, corpus_vecs


def get_hasher(method, dim, kc=2000, sparsity=0.05):
    global hashers
    key = f"{method}_{dim}_{kc}_{sparsity}"
    if key not in hashers:
        if method == "flyhash":
            hashers[key] = FlyHash(input_dim=dim, kenyon_cells=kc, sparsity=sparsity, seed=42)
        elif method == "learned":
            _, X = get_corpus()
            h = LearnedFlyHash(input_dim=dim, kenyon_cells=kc, sparsity=sparsity,
                               seed=42, learning_rate=0.005, n_epochs=3)
            h.fit(X)
            hashers[key] = h
        elif method == "connectome":
            hashers[key] = ConnectomeFlyHash(input_dim=dim, kenyon_cells=kc,
                                             sparsity=sparsity, seed=42)
        else:
            hashers[key] = FlyHash(input_dim=dim, kenyon_cells=kc, sparsity=sparsity, seed=42)
    return hashers[key]


def get_fps(method, kc=2000, sparsity=0.05):
    global corpus_fps
    _, X = get_corpus()
    key = f"{method}_{kc}_{sparsity}"
    if key not in corpus_fps:
        h = get_hasher(method, X.shape[1], kc, sparsity)
        corpus_fps[key] = h.hash_batch(X)
    return corpus_fps[key]


# ── Models ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    method: str = "flyhash"
    top_k: int = 5
    kenyon_cells: int = 2000
    sparsity: float = 0.05


class BenchmarkRequest(BaseModel):
    top_k: int = 5
    n_queries: int = 20
    match: str = "bits"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/corpus")
def api_corpus():
    texts, _ = get_corpus()
    return {"corpus": texts, "count": len(texts)}


@app.post("/api/search")
def api_search(req: SearchRequest):
    texts, X = get_corpus()
    sp = req.sparsity
    kc = req.kenyon_cells

    hasher = get_hasher(req.method, X.shape[1], kc, sp)
    fps = get_fps(req.method, kc, sp)

    q_vec = get_embedder().encode([req.query])[0].astype(np.float64)
    q_fp = hasher.hash(q_vec)
    idx, scores = hasher.find_neighbors(q_vec, fps, top_k=req.top_k)

    # Kenyon cell activation grid
    display_kc = min(kc, 2000)
    grid = q_fp[:display_kc].tolist()
    bits_set = int(q_fp.sum())

    results = []
    for i, s in zip(idx.tolist(), scores.tolist()):
        results.append({"index": i, "text": texts[i], "score": round(s, 4)})

    return {
        "results": results,
        "kenyon_grid": grid,
        "bits_set": bits_set,
        "kenyon_cells": kc,
        "sparsity_actual": round(bits_set / kc * 100, 1),
        "method": req.method,
    }


@app.post("/api/compare")
def api_compare(req: SearchRequest):
    """Search with all 3 fly variants at once."""
    texts, X = get_corpus()
    sp = req.sparsity
    kc = req.kenyon_cells

    q_vec = get_embedder().encode([req.query])[0].astype(np.float64)

    all_results = {}
    for method in ["flyhash", "learned", "connectome"]:
        hasher = get_hasher(method, X.shape[1], kc, sp)
        fps = get_fps(method, kc, sp)
        idx, scores = hasher.find_neighbors(q_vec, fps, top_k=req.top_k)
        all_results[method] = [
            {"index": int(i), "text": texts[i], "score": round(float(s), 4)}
            for i, s in zip(idx, scores)
        ]

    return {"variants": all_results}


@app.post("/api/benchmark")
def api_benchmark(req: BenchmarkRequest):
    _, X = get_corpus()
    results = run_benchmark(
        X, labels=None,
        top_k=req.top_k,
        n_queries=req.n_queries,
        match=req.match,
    )
    rows = []
    for name, vals in results.items():
        rows.append({
            "method": name,
            "recall": round(vals["recall"], 3),
            "purity": round(vals["purity"], 3) if vals.get("purity") is not None else None,
            "qps": round(vals["qps"]),
            "memory_kb": round(vals["memory_kb"], 1),
        })
    return {"results": rows}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0"}
