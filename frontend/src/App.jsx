import { useState } from "react"
import FlyGame from "./FlyGame"

const API = ""

function App() {
  const [view, setView] = useState("landing")
  const [tab, setTab] = useState("tutorial")
  const [method, setMethod] = useState("flyhash")
  const [query, setQuery] = useState("")
  const [results, setResults] = useState(null)
  const [searchMeta, setSearchMeta] = useState(null)
  const [loading, setLoading] = useState(false)
  const [benchResults, setBenchResults] = useState(null)
  const [benchLoading, setBenchLoading] = useState(false)
  const [compareResults, setCompareResults] = useState(null)
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareQuery, setCompareQuery] = useState("")

  const doSearch = async () => {
    if (!query.trim()) return
    setLoading(true)
    try {
      const res = await fetch(`${API}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim(), method, top_k: 5 }),
      })
      const data = await res.json()
      setResults(data.results)
      setSearchMeta(data)
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  const doBenchmark = async () => {
    setBenchLoading(true)
    try {
      const res = await fetch(`${API}/api/benchmark`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ top_k: 5, n_queries: 20, match: "bits" }),
      })
      const data = await res.json()
      setBenchResults(data.results)
    } catch (e) { console.error(e) }
    setBenchLoading(false)
  }

  const doCompare = async () => {
    if (!compareQuery.trim()) return
    setCompareLoading(true)
    try {
      const res = await fetch(`${API}/api/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: compareQuery.trim(), top_k: 5 }),
      })
      const data = await res.json()
      setCompareResults(data.variants)
    } catch (e) { console.error(e) }
    setCompareLoading(false)
  }

  const methodClass = (m) => {
    if (method !== m) return "method-chip"
    if (m === "flyhash") return "method-chip active-fly"
    if (m === "learned") return "method-chip active-learned"
    return "method-chip active-conn"
  }

  const methodColors = {
    "FlyHash (random W)": "var(--accent-teal)",
    "LearnedFly (Hebbian)": "var(--accent-red)",
    "Connectome (bio-wiring)": "var(--accent-orange)",
    "ClassicLSH (baseline)": "var(--accent-blue)",
    "PCAHash (data-driven)": "var(--accent-purple)",
  }

  return (
    <div className="app-container">

      {view === "landing" && (
        <div className="landing-page">
          <div className="landing-hero">
            <div className="landing-badge">🧬 Bio-Inspired · Drosophila Connectome</div>
            <h1 className="landing-logo">Smell-Alike OS</h1>
            <p className="landing-tagline">
              Two experiences powered by the real fruit fly brain circuit.
            </p>
          </div>
          <div className="landing-cards">
            <div className="landing-card" onClick={() => setView("app")}>
              <span className="landing-card-icon">🔍</span>
              <h3>Fly-Brain Search</h3>
              <p>Similarity search using the mushroom body LSH algorithm (Dasgupta 2017).</p>
              <span className="landing-card-tag">Science App</span>
            </div>
            <div className="landing-card" onClick={() => setView("game")}>
              <span className="landing-card-icon">🪰</span>
              <h3>Speedrun Game</h3>
              <p>A fly that learns from crashes using real connectome neural weights.</p>
              <span className="landing-card-tag">AI Game</span>
            </div>
          </div>
        </div>
      )}
      {view === "game" && <FlyGame onBack={() => setView("landing")} />}
      {view === "app" && (
        <>
          <button className="back-btn top-back" onClick={() => setView("landing")}>← Menu</button>

      {/* ── Hero ──────────────────────────────────── */}
      <section className="hero">
        <div className="hero-badge">🧬 Bio-Inspired Algorithm</div>
        <h1>Smell-Alike</h1>
        <p className="hero-subtitle">
          A similarity search engine running a fruit fly's neural algorithm.
          Expand. Fire. Sparsify.
        </p>
      </section>

      {/* ── Navigation ────────────────────────────── */}
      <nav className="nav-tabs">
        {[
          ["tutorial", "📖 Tutorial"],
          ["search", "🔍 Search"],
          ["compare", "⚔️ Compare"],
          ["benchmark", "📊 Benchmark"],
          ["how", "🧬 Science"],
        ].map(([id, label]) => (
          <button
            key={id}
            className={`nav-tab ${tab === id ? "active" : ""}`}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      {/* ═══════════════ TUTORIAL TAB ════════════════ */}
      {tab === "tutorial" && (
        <section>
          <h2 className="section-title">Welcome to Smell-Alike 🪰</h2>
          <p className="section-subtitle">
            A quick guide to get you started. This app searches your notes
            using an algorithm discovered in a fruit fly's brain.
          </p>

          <div className="tutorial-steps">
            <div className="tutorial-step">
              <div className="tutorial-number">1</div>
              <div className="tutorial-content">
                <h3>🔍 Search for similar notes</h3>
                <p>
                  Go to the <strong>Search</strong> tab, type any question or topic,
                  and hit Search. The fly algorithm will find the most similar notes
                  in the corpus by creating binary "smell fingerprints" and matching overlaps.
                </p>
                <div className="tutorial-tip">
                  💡 Try: "How does memory work in the brain?" or "What is quantum computing?"
                </div>
                <button className="try-btn" onClick={() => { setTab("search"); setQuery("How does memory work in the brain?"); }}>
                  → Try it now
                </button>
              </div>
            </div>

            <div className="tutorial-step">
              <div className="tutorial-number">2</div>
              <div className="tutorial-content">
                <h3>🧬 Choose a fly brain variant</h3>
                <p>
                  Before searching, pick which type of fly brain to use. Each variant
                  uses a different projection matrix — the "wiring" between neurons:
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 12 }}>
                  <div className="tutorial-tip" style={{ background: "var(--accent-teal-dim)", color: "var(--accent-teal)" }}>
                    🟢 <strong>Random W</strong> — Original paper. Random sparse wiring, like a newborn fly.
                  </div>
                  <div className="tutorial-tip" style={{ background: "rgba(255,107,107,0.1)", color: "var(--accent-red)" }}>
                    🔴 <strong>Hebbian</strong> — Learned wiring. Trained on your corpus, like an adult fly.
                  </div>
                  <div className="tutorial-tip" style={{ background: "rgba(255,159,67,0.1)", color: "var(--accent-orange)" }}>
                    🟠 <strong>Connectome</strong> — Bio-realistic wiring from real fly anatomy data.
                  </div>
                </div>
              </div>
            </div>

            <div className="tutorial-step">
              <div className="tutorial-number">3</div>
              <div className="tutorial-content">
                <h3>⚔️ Compare all variants at once</h3>
                <p>
                  Go to the <strong>Compare</strong> tab, type one query, and see how
                  all three fly brain variants rank the results differently — side by side.
                  This shows how different wiring patterns affect similarity perception.
                </p>
                <button className="try-btn" onClick={() => { setTab("compare"); setCompareQuery("evolution and natural selection"); }}>
                  → Try comparing
                </button>
              </div>
            </div>

            <div className="tutorial-step">
              <div className="tutorial-number">4</div>
              <div className="tutorial-content">
                <h3>📊 Run the benchmark</h3>
                <p>
                  Go to the <strong>Benchmark</strong> tab and click "Run Benchmark" to
                  see a head-to-head comparison of all 5 methods (3 fly variants + 2
                  classical baselines). The table shows Recall@5, queries per second,
                  and memory usage.
                </p>
                <div className="tutorial-tip">
                  💡 Spoiler: the fly's algorithm beats classical LSH by +47% recall!
                </div>
                <button className="try-btn" onClick={() => setTab("benchmark")}>
                  → See the benchmark
                </button>
              </div>
            </div>

            <div className="tutorial-step">
              <div className="tutorial-number">5</div>
              <div className="tutorial-content">
                <h3>🧬 Understand the science</h3>
                <p>
                  The <strong>Science</strong> tab explains the 3-step algorithm
                  (Expand → Fire → Sparsify) and why the fly's approach works better
                  than traditional computer science methods.
                </p>
                <button className="try-btn" onClick={() => setTab("how")}>
                  → Learn the science
                </button>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ═══════════════ SEARCH TAB ═════════════════ */}
      {tab === "search" && (
        <section>
          <div className="method-selector">
            {[["flyhash","🟢 Random W"],["learned","🔴 Hebbian"],["connectome","🟠 Connectome"]].map(([m,label]) => (
              <button key={m} className={methodClass(m)} onClick={() => setMethod(m)}>{label}</button>
            ))}
          </div>

          <div className="search-container">
            <input className="search-input" type="text"
              placeholder="How do neural circuits implement algorithms?"
              value={query} onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && doSearch()} />
            <button className="search-btn" onClick={doSearch} disabled={loading || !query.trim()}>
              {loading ? <><span className="spinner"/>Searching...</> : "🪰 Search"}
            </button>
          </div>

          {searchMeta && (
            <>
              <div className="stats-row">
                <div className="stat-card">
                  <div className="stat-value">{searchMeta.bits_set}</div>
                  <div className="stat-label">Cells Fired</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{searchMeta.sparsity_actual}%</div>
                  <div className="stat-label">Sparsity</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{searchMeta.method}</div>
                  <div className="stat-label">Variant</div>
                </div>
              </div>

              <div className="heatmap-container">
                <div className="heatmap-title">
                  🧬 Kenyon Cell Activation — {searchMeta.bits_set} / {searchMeta.kenyon_cells} fired
                </div>
                <div className="heatmap-grid">
                  {searchMeta.kenyon_grid.slice(0, 2000).map((v, i) => (
                    <div key={i} className="heatmap-cell" style={{
                      background: v ? (method === "learned" ? "var(--accent-red)"
                        : method === "connectome" ? "var(--accent-orange)" : "var(--accent-teal)")
                        : "rgba(255,255,255,0.02)"
                    }} />
                  ))}
                </div>
              </div>

              <div className="bio-explainer">
                🧬 <strong>What happened:</strong> Your query → 384-D embedding →
                expanded to <strong>{searchMeta.kenyon_cells} Kenyon cells</strong> →
                winner-take-all kept top <strong>{searchMeta.sparsity_actual}%
                ({searchMeta.bits_set} cells)</strong> → binary fingerprint →
                matched against corpus fingerprints.
              </div>
            </>
          )}

          {results && (
            <div className="results-grid">
              {results.map((r, i) => (
                <div key={i} className="result-item" style={{ animationDelay: `${i * 0.08}s` }}>
                  <div className="result-rank">{i + 1}</div>
                  <div className="result-body">
                    <div className="result-text">{r.text}</div>
                    <div className="result-score">Jaccard similarity: {r.score}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* ═══════════════ COMPARE TAB ════════════════ */}
      {tab === "compare" && (
        <section>
          <h2 className="section-title">Side-by-Side Comparison</h2>
          <p className="section-subtitle">
            Same query, three different fly brains — see which finds better matches.
          </p>
          <div className="search-container">
            <input className="search-input" placeholder="memory and learning in the brain"
              value={compareQuery} onChange={(e) => setCompareQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && doCompare()} />
            <button className="search-btn" onClick={doCompare} disabled={compareLoading || !compareQuery.trim()}>
              {compareLoading ? <><span className="spinner"/>Comparing...</> : "⚔️ Compare"}
            </button>
          </div>
          {compareResults && (
            <div className="compare-grid">
              {[["flyhash","🟢 Random W"],["learned","🔴 Hebbian"],["connectome","🟠 Connectome"]].map(([key,title]) => (
                <div key={key} className="compare-column">
                  <h3>{title}</h3>
                  {compareResults[key]?.map((r, i) => (
                    <div key={i} className="compare-item">
                      <strong>#{i+1}</strong> {r.text}
                      <div className="score">Jaccard: {r.score}</div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* ═══════════════ BENCHMARK TAB ══════════════ */}
      {tab === "benchmark" && (
        <section>
          <h2 className="section-title">5-Method Benchmark</h2>
          <p className="section-subtitle">
            Head-to-head: 3 fly variants vs 2 classical baselines. Recall@5, speed, and memory.
          </p>
          <button className="bench-run-btn" onClick={doBenchmark} disabled={benchLoading}>
            {benchLoading ? <><span className="spinner"/>Running...</> : "▶ Run Benchmark"}
          </button>
          {benchResults && (
            <table className="bench-table">
              <thead><tr><th>Method</th><th>Recall@5</th><th style={{width:"30%"}}>Visual</th><th>QPS</th><th>Memory</th></tr></thead>
              <tbody>
                {benchResults.map((r, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 600 }}>{r.method}</td>
                    <td style={{ fontFamily: "var(--font-mono)", color: methodColors[r.method] }}>{r.recall}</td>
                    <td><div className="bench-bar" style={{ width: `${r.recall * 100}%`, background: methodColors[r.method] || "var(--accent-teal)" }} /></td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{r.qps}</td>
                    <td style={{ fontFamily: "var(--font-mono)" }}>{r.memory_kb} KB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      )}

      {/* ═══════════════ HOW IT WORKS ═══════════════ */}
      {tab === "how" && (
        <section>
          <h2 className="section-title">The Fly's Algorithm</h2>
          <p className="section-subtitle">
            In 2017, scientists discovered that fruit flies run a better version
            of locality-sensitive hashing — the core algorithm behind search engines.
          </p>
          <div className="steps-grid">
            <div className="step-card">
              <div className="step-icon">🔀</div>
              <div className="step-title">1. Expand</div>
              <div className="step-desc">
                Instead of compressing, the fly <strong>expands</strong>.
                384 inputs project to 2,000 Kenyon cells via sparse wiring (6%).
              </div>
              <div className="step-detail">384D → 2,000 KCs</div>
            </div>
            <div className="step-card">
              <div className="step-icon">⚡</div>
              <div className="step-title">2. Fire</div>
              <div className="step-desc">
                Each Kenyon cell computes a weighted sum. Sparse connectivity means
                each cell sees only ~23 of the 384 features.
              </div>
              <div className="step-detail">activation = W · x</div>
            </div>
            <div className="step-card">
              <div className="step-icon">🏆</div>
              <div className="step-title">3. Sparsify</div>
              <div className="step-desc">
                Winner-take-all: only top 5% survive. That sparse binary pattern
                is the fingerprint. Similar inputs → overlapping prints.
              </div>
              <div className="step-detail">keep top 5%</div>
            </div>
          </div>
          <div className="bio-explainer">
            <strong>Why expansion works:</strong> Two similar smells activating slightly different
            receptors get amplified into clearly distinct patterns across 2,000 cells — but
            similar inputs still share most active cells. The result: <strong>+47% more true
            neighbours found</strong> than classical LSH. Evolution beat computer science
            by billions of years.
          </div>
        </section>
      )}

      <footer className="footer">
        <p>Smell-Alike v2 · Based on <a href="https://doi.org/10.1126/science.aam8084" target="_blank" rel="noopener">Dasgupta, Stevens & Navlakha (2017)</a></p>
      </footer>
        </>
      )}
    </div>
  )
}

export default App
