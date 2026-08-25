"""
streamlit_app.py  --  Smell-Alike v2 UI
========================================
A similarity search engine secretly running a fruit fly's nose.
Now featuring: Hebbian learning, connectome wiring, and 5-method benchmarks.

Run:  streamlit run app/streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go

from smell_alike.fly_hash import FlyHash
from smell_alike.learned_fly_hash import LearnedFlyHash
from smell_alike.connectome import ConnectomeFlyHash
from smell_alike.embeddings import TextEmbedder, load_sample_corpus, load_notes_corpus
from smell_alike.benchmark import run_benchmark

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smell-Alike v2 🪰",
    page_icon="🪰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e0e6f0;
}
.main-header {
    text-align: center; padding: 2rem 0 1rem;
    background: linear-gradient(135deg, rgba(0,200,180,0.08), rgba(100,120,255,0.08));
    border-radius: 16px; border: 1px solid rgba(0,200,180,0.15); margin-bottom: 2rem;
}
.main-header h1 {
    font-size: 3rem; font-weight: 700;
    background: linear-gradient(90deg, #00c8b4, #6478ff, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}
.main-header p { color: #8892a4; font-size: 1.1rem; margin-top: 0.5rem; }
.version-badge {
    display: inline-block; background: linear-gradient(135deg, #ff6b6b, #c084fc);
    color: white; font-size: 0.7rem; font-weight: 700; padding: 2px 8px;
    border-radius: 8px; margin-left: 8px; vertical-align: super;
}

.result-card {
    background: linear-gradient(135deg, rgba(15,25,45,0.9), rgba(20,30,55,0.9));
    border: 1px solid rgba(0,200,180,0.2); border-radius: 12px;
    padding: 1rem 1.2rem; margin-bottom: 0.75rem; transition: all 0.2s ease;
}
.result-card:hover { border-color: rgba(0,200,180,0.5); transform: translateX(4px); }
.result-rank { font-size: 0.75rem; color: #00c8b4; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em; }
.result-text { font-size: 1rem; color: #d0d8e8; margin: 0.3rem 0; }
.result-score { font-size: 0.8rem; color: #6478ff; font-weight: 500; }

.bio-box {
    background: linear-gradient(135deg, rgba(0,200,180,0.05), rgba(100,120,255,0.05));
    border: 1px solid rgba(0,200,180,0.25); border-radius: 12px;
    padding: 1rem 1.4rem; margin: 1rem 0; font-size: 0.9rem;
    color: #9aa8c0; line-height: 1.7;
}
.bio-box strong { color: #00c8b4; }
.stat-box {
    background: rgba(20,30,55,0.8); border: 1px solid rgba(100,120,255,0.2);
    border-radius: 10px; padding: 0.8rem; text-align: center;
}
.stat-value { font-size: 1.6rem; font-weight: 700; color: #6478ff; }
.stat-label { font-size: 0.75rem; color: #6e7a8a; margin-top: 0.2rem; }

.method-tag {
    display: inline-block; padding: 2px 10px; border-radius: 6px;
    font-size: 0.75rem; font-weight: 600; margin-right: 6px;
}
.method-fly    { background: rgba(0,200,180,0.2); color: #00c8b4; }
.method-learned { background: rgba(255,107,107,0.2); color: #ff6b6b; }
.method-conn   { background: rgba(255,165,0,0.2); color: #ffa500; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading sentence encoder...")
def load_embedder():
    return TextEmbedder()

@st.cache_data(show_spinner="Encoding corpus...")
def encode_corpus(texts):
    emb = load_embedder()
    return emb.encode(list(texts), show_progress=False).astype(np.float64)

@st.cache_resource
def get_hasher(method_name, input_dim, kenyon_cells, sparsity, _corpus_hash=None, _X=None):
    """Build the selected hasher. _corpus_hash is used for cache invalidation."""
    if method_name == "FlyHash (random W)":
        return FlyHash(input_dim=input_dim, kenyon_cells=kenyon_cells,
                       sparsity=sparsity, seed=42)
    elif method_name == "LearnedFly (Hebbian)":
        lf = LearnedFlyHash(input_dim=input_dim, kenyon_cells=kenyon_cells,
                            sparsity=sparsity, seed=42,
                            learning_rate=0.005, n_epochs=3)
        if _X is not None:
            lf.fit(_X)
        return lf
    elif method_name == "Connectome (bio-wiring)":
        return ConnectomeFlyHash(input_dim=input_dim, kenyon_cells=kenyon_cells,
                                 sparsity=sparsity, seed=42)
    else:
        return FlyHash(input_dim=input_dim, kenyon_cells=kenyon_cells,
                       sparsity=sparsity, seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🪰 Smell-Alike <span class="version-badge">v2</span></h1>
  <p>A similarity search engine running a fruit fly's nose — now with Hebbian learning</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    corpus_source = st.radio(
        "Corpus source",
        ["Built-in sample (60 notes)", "Upload your own .txt file"],
    )
    uploaded = None
    if corpus_source == "Upload your own .txt file":
        uploaded = st.file_uploader("One note per line", type=["txt"])

    st.divider()
    st.markdown("### 🧬 Fly parameters")
    hasher_choice = st.selectbox(
        "Hasher variant",
        ["FlyHash (random W)", "LearnedFly (Hebbian)", "Connectome (bio-wiring)"],
        help="Random = original paper. Learned = Hebbian-trained W. Connectome = bio-realistic wiring."
    )
    kenyon_cells = st.slider("Kenyon cells (expansion size)", 500, 4000, 2000, 100)
    sparsity_pct = st.slider("Sparsity % (WTA survivors)", 1, 20, 5, 1)
    top_k = st.slider("Results to show", 1, 10, 5, 1)

    st.divider()
    st.markdown("""
### 📖 How it works
**3 steps — like the fly:**

1. **Expand** — 384D embedding → 2000 Kenyon cells via sparse projection
2. **Fire** — each KC computes a weighted sum of inputs
3. **Sparsify** — WTA keeps top 5% → binary fingerprint

**v2 variants:**
- 🟢 **Random W** — original Dasgupta et al. (2017)
- 🔴 **Hebbian** — learned projection (trained fly)
- 🟠 **Connectome** — bio-realistic log-normal wiring

[2017 Science paper](https://doi.org/10.1126/science.aam8084)
""")

# ─────────────────────────────────────────────────────────────────────────────
# Load corpus
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is not None:
    raw = uploaded.read().decode("utf-8")
    corpus = [l.strip() for l in raw.splitlines() if l.strip()]
    st.toast(f"Loaded {len(corpus)} notes from file", icon="📄")
else:
    corpus = load_sample_corpus()

corpus_tuple = tuple(corpus)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_search, tab_compare, tab_bench, tab_corpus = st.tabs(
    ["🔍 Search", "⚔️ Compare", "📊 Benchmark", "📚 Corpus"]
)

# ═══════════════════════════ TAB 1 — SEARCH ═══════════════════════════════════
with tab_search:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### ✏️ Your query")
        query_text = st.text_area(
            "Query", height=120,
            placeholder="e.g.  How do neural circuits implement algorithms?",
            label_visibility="collapsed",
        )
        # Show which method is active
        method_class = {"FlyHash (random W)": "method-fly",
                        "LearnedFly (Hebbian)": "method-learned",
                        "Connectome (bio-wiring)": "method-conn"}[hasher_choice]
        st.markdown(
            f'<span class="method-tag {method_class}">{hasher_choice}</span>',
            unsafe_allow_html=True,
        )
        search_btn = st.button("🪰 Find similar notes", type="primary", use_container_width=True)

    with col_right:
        st.markdown("### 🧬 Kenyon cell activation")
        cell_placeholder = st.empty()
        cell_placeholder.info("Run a search to see which Kenyon cells fire.")

    if search_btn and query_text.strip():
        with st.spinner("Searching..."):
            X = encode_corpus(corpus_tuple)
            corpus_hash = hash(corpus_tuple)
            hasher = get_hasher(hasher_choice, X.shape[1], kenyon_cells,
                                sparsity_pct / 100.0,
                                _corpus_hash=corpus_hash, _X=X)
            corpus_fps = hasher.hash_batch(X)
            q_vec = load_embedder().encode([query_text.strip()])[0].astype(np.float64)
            t0 = time.perf_counter()
            q_fp = hasher.hash(q_vec)
            idx, scores = hasher.find_neighbors(q_vec, corpus_fps, top_k=top_k)
            search_ms = (time.perf_counter() - t0) * 1000

        # Kenyon cell heatmap
        display_kc = min(kenyon_cells, 2000)
        grid_h, grid_w = 40, 50
        grid = q_fp[:display_kc].reshape(grid_h, grid_w)
        heatmap_color = {"FlyHash (random W)": "#00c8b4",
                         "LearnedFly (Hebbian)": "#ff6b6b",
                         "Connectome (bio-wiring)": "#ffa500"}[hasher_choice]
        fig = go.Figure(go.Heatmap(
            z=grid.astype(float),
            colorscale=[[0, "#0a0e1a"], [1, heatmap_color]],
            showscale=False,
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0), height=200,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        cell_placeholder.plotly_chart(fig, use_container_width=True)

        bits_set = int(q_fp.sum())
        c1, c2, c3 = st.columns(3)
        for col, val, label in [
            (c1, str(bits_set), "Kenyon cells fired"),
            (c2, f"{bits_set/kenyon_cells*100:.1f}%", "Actual sparsity"),
            (c3, f"{search_ms:.1f}ms", "Search time"),
        ]:
            with col:
                st.markdown(
                    f'<div class="stat-box"><div class="stat-value">{val}</div>' +
                    f'<div class="stat-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

        variant_desc = {
            "FlyHash (random W)": "a <strong>random sparse projection</strong> matrix (original paper)",
            "LearnedFly (Hebbian)": "a <strong>Hebbian-learned projection</strong> matrix (trained on your corpus)",
            "Connectome (bio-wiring)": "a <strong>connectome-derived projection</strong> with log-normal synaptic weights",
        }[hasher_choice]

        st.markdown(f"""
<div class="bio-box">
🧬 <strong>What just happened (biologically):</strong><br>
Your query was encoded as a 384-D sentence embedding, then expanded to
<strong>{kenyon_cells} Kenyon cells</strong> via {variant_desc}
(6% connectivity — matching the real fly's PN→KC wiring density). A winner-take-all
inhibitory neuron silenced all but the <strong>top {sparsity_pct}% ({bits_set} cells)</strong>.
That surviving pattern is the fingerprint. Similar notes share overlapping fingerprints.
</div>""", unsafe_allow_html=True)

        st.markdown("### 🎯 Most similar notes")
        for rank, (i, score) in enumerate(zip(idx, scores), 1):
            st.markdown(
                f'<div class="result-card">' +
                f'<div class="result-rank">#{rank}</div>' +
                f'<div class="result-text">{corpus[i]}</div>' +
                f'<div class="result-score">Jaccard similarity: {score:.4f}</div>' +
                f'</div>',
                unsafe_allow_html=True,
            )

    elif search_btn:
        st.warning("Please enter a query first.")


# ═══════════════════════════ TAB 2 — COMPARE ══════════════════════════════════
with tab_compare:
    st.markdown("### ⚔️ Side-by-side: all 3 fly variants")
    st.markdown("Enter a query and see how Random, Learned, and Connectome rank results differently.")

    compare_query = st.text_input(
        "Query for comparison",
        placeholder="e.g.  memory and learning in the brain",
    )
    compare_btn = st.button("⚔️ Compare all variants", type="primary", use_container_width=True)

    if compare_btn and compare_query.strip():
        with st.spinner("Running all 3 variants..."):
            X = encode_corpus(corpus_tuple)
            q_vec = load_embedder().encode([compare_query.strip()])[0].astype(np.float64)
            sp = sparsity_pct / 100.0

            variants = {
                "🟢 FlyHash (random)": FlyHash(input_dim=X.shape[1], kenyon_cells=kenyon_cells, sparsity=sp, seed=42),
                "🔴 LearnedFly (Hebbian)": LearnedFlyHash(input_dim=X.shape[1], kenyon_cells=kenyon_cells, sparsity=sp, seed=42, learning_rate=0.005, n_epochs=3),
                "🟠 Connectome (bio)": ConnectomeFlyHash(input_dim=X.shape[1], kenyon_cells=kenyon_cells, sparsity=sp, seed=42),
            }
            # Fit the learned variant
            variants["🔴 LearnedFly (Hebbian)"].fit(X)

        cols = st.columns(3)
        for col, (name, method) in zip(cols, variants.items()):
            with col:
                st.markdown(f"**{name}**")
                fps = method.hash_batch(X)
                idx_v, scores_v = method.find_neighbors(q_vec, fps, top_k=top_k)
                for rank, (i, score) in enumerate(zip(idx_v, scores_v), 1):
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="result-rank">#{rank} · {score:.3f}</div>'
                        f'<div class="result-text" style="font-size:0.85rem">{corpus[i]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    elif compare_btn:
        st.warning("Please enter a query first.")


# ══════════════════════════ TAB 3 — BENCHMARK ═════════════════════════════════
with tab_bench:
    st.markdown("### 📊 5-method head-to-head")
    st.markdown("""
All fly variants + classical baselines on the loaded corpus:
- **Recall@K** — fraction of true top-K neighbours found (↑ better)
- **Purity@K** — topical clustering quality (↑ better, when labels available)
- **QPS** — queries per second (↑ better)
- **Memory KB** — fingerprint storage (↓ better)
""")
    bench_k       = st.number_input("K for Recall@K", 1, 20, 5)
    bench_queries = st.number_input("Query samples", 5, 50, 20)
    match_mode    = st.radio(
        "Baseline code-length matching",
        ["bits — same # of 1s as FlyHash (fair fight)",
         "storage — same code length as FlyHash"],
    )
    bench_bits    = st.number_input(
        "Baseline code bits (storage mode)", 10, 4000, 2000,
        disabled=match_mode.startswith("bits"),
    )
    st.caption("PCAHash automatically caps its bits at N-1 (corpus size).")

    if st.button("▶ Run benchmark", type="primary"):
        with st.spinner("Running 5-method benchmark (may take ~30s)..."):
            X = encode_corpus(corpus_tuple).astype(np.float64)
            results = run_benchmark(
                X, labels=None,
                top_k=int(bench_k),
                n_queries=int(bench_queries),
                match="bits" if match_mode.startswith("bits") else "storage",
                baseline_bits=int(bench_bits),
            )

        import pandas as pd
        def _row(m, v):
            row = {"Method": m,
                   f"Recall@{int(bench_k)}": f"{v['recall']:.3f}"}
            if v.get("purity") is not None:
                row[f"Purity@{int(bench_k)}"] = f"{v['purity']:.3f}"
            row["QPS"] = f"{v['qps']:.0f}"
            row["Memory KB"] = f"{v['memory_kb']:.1f}"
            return row
        rows = [_row(m, v) for m, v in results.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        colors = ["#00c8b4", "#ff6b6b", "#ffa500", "#6478ff", "#c084fc"]
        fig = go.Figure(go.Bar(
            x=list(results.keys()),
            y=[results[m]["recall"] for m in results],
            marker_color=colors[:len(results)],
            text=[f"{results[m]['recall']:.3f}" for m in results],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Recall@{int(bench_k)} comparison",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,14,26,0.8)",
            font_color="#d0d8e8",
            yaxis=dict(range=[0, 1.15], gridcolor="rgba(255,255,255,0.05)"),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════ TAB 4 — CORPUS ════════════════════════════════════
with tab_corpus:
    st.markdown(f"### 📚 Corpus ({len(corpus)} items)")
    st.caption("Upload your own notes via the sidebar to replace this.")
    filt = st.text_input("Filter:", placeholder="type to search...")
    filtered = [(i, c) for i, c in enumerate(corpus) if filt.lower() in c.lower()] if filt else list(enumerate(corpus))
    for idx_n, note in filtered:
        st.markdown(
            f'<div class="result-card">' +
            f'<span style="color:#6e7a8a;font-size:0.75rem">#{idx_n}</span> ' +
            f'<span class="result-text">{note}</span></div>',
            unsafe_allow_html=True,
        )
