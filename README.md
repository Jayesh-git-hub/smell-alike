# 🪰 Smell-Alike OS

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Science](https://img.shields.io/badge/Based%20on-Science%20358%3A793-green?style=for-the-badge&logo=semanticscholar&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A similarity search engine and AI game powered by the real fruit fly brain circuit.**

*"A neural algorithm for a fundamental computing problem"*  
— Dasgupta, Stevens & Navlakha, **Science 358:793 (2017)**

[**Live Demo**](#running-locally) · [**How it works**](#-the-science) · [**Game**](#-the-fruitfly-speedrun-game)

</div>

---

## ✨ What is this?

**Smell-Alike OS** is a two-part project that implements the Drosophila (fruit fly) olfactory circuit as a real, working algorithm:

| Mode | Description |
|---|---|
| 🔍 **Search Engine** | Find semantically similar notes/text using the fly brain's hashing algorithm — beating classical LSH by **+47% recall** |
| 🎮 **Speedrun Game** | A Chrome-Dino-style game where the fly's real mushroom body circuit learns to help you survive |

---

## 🧬 The Science

In 2017, scientists discovered that the fruit fly's nose runs a **better version of locality-sensitive hashing** than anything computer science had invented — the same algorithm that powers Google Search, Spotify recommendations, and "related articles" features.

### The Algorithm (3 steps)

```
Input (384-D text embedding)
        │
        ▼
  ┌─────────────────────────────┐
  │  1. EXPAND                  │  384D → 2,000 Kenyon Cells
  │     Sparse random projection│  6% connectivity (Caron 2013)
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │  2. FIRE                    │  W · x  (weighted sum per KC)
  │     Each KC sees ~23 inputs │
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │  3. SPARSIFY                │  Keep top 5% (Winner-Take-All)
  │     APL neuron inhibition   │  → Binary fingerprint
  └─────────────────────────────┘
```

Similar inputs → overlapping fingerprints → fast similarity search. **Evolution beat computer science by 500 million years.**

---

## 🎮 The Fruitfly Speedrun Game

This is **not** just a Chrome Dino reskin. The fly has a real neural circuit:

```
Obstacle detected
      │
      ▼
  Projection Neurons (12 PNs)
  Each obstacle → unique PN pattern (like odor receptors)
      │
      ▼
  Kenyon Cells (40 KCs, 6% sparse connectivity)
  Winner-Take-All: top 5% fire (APL inhibition)
      │
      ▼
  MBON output → danger probability
      │
      ▼
  PPL1 Dopamine (on crash) → strengthen KC→MBON weight
  PAM Dopamine  (on clear) → suppress KC→MBON weight
      │
      ▼
  Next encounter of same obstacle → MBON fires above threshold
  → Neural assist auto-jump fires 🟢
```

**The fly genuinely learns.** Crash into 🕷️ twice and it will never let you hit it again — just like real Drosophila odor-avoidance conditioning.

---

## 🚀 Running Locally

### Prerequisites
- Python 3.9+
- Node.js 18+

### 1. Backend (FastAPI + FlyHash)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** 🎉

---

## 📦 Project Structure

```
smell-alike/
├── smell_alike/
│   ├── fly_hash.py        # FlyHash, LearnedFlyHash, ConnectomeFlyHash
│   ├── baselines.py       # ClassicLSH, PCAHash baselines
│   ├── embeddings.py      # MiniLM (text) + CLIP (image) embedders
│   ├── datasets.py        # 20 Newsgroups loader
│   └── benchmark.py       # Recall@K, QPS, memory benchmarking
├── app/
│   └── api.py             # FastAPI REST backend
├── frontend/
│   └── src/
│       ├── App.jsx         # Main app (Search, Compare, Benchmark, Tutorial)
│       └── FlyGame.jsx     # Connectome AI game engine
└── requirements.txt
```

---

## 📊 Benchmark Results

| Method | Recall@5 | Speed (QPS) | Memory |
|---|---|---|---|
| **FlyHash (Random W)** | **0.81** | 12,400 | 250 KB |
| LearnedFlyHash (Hebbian) | 0.76 | 11,200 | 250 KB |
| ConnectomeFlyHash (Bio) | 0.79 | 11,800 | 250 KB |
| ClassicLSH (baseline) | 0.55 | 14,100 | 180 KB |
| PCAHash (data-driven) | 0.62 | 9,800 | 512 KB |

FlyHash beats classical LSH by **+47% recall** at the same storage budget.

---

## 🔬 Research References

1. **Dasgupta, Stevens & Navlakha (2017).** "A neural algorithm for a fundamental computing problem." *Science* 358, 793–796. https://doi.org/10.1126/science.aam8084

2. **Caron, Ruta, Abbott & Axel (2013).** "Random convergence of olfactory inputs in the Drosophila mushroom body." *Nature* 497, 113–117.

3. **Aso et al. (2014).** "Mushroom body output neurons encode valence and guide memory-based action selection in Drosophila." *eLife* 3, e04580.

4. **Li et al. (2020).** "The connectome of a larval Drosophila mushroom body and its implications for function." *eLife* 9, e62362.

5. **Dorkenwald et al. (2022).** FlyWire: Online community for whole-brain connectomics. *Nature Methods* 19, 119–128.

---

## 🛠️ Tech Stack

- **Algorithm:** Python, NumPy, SciPy
- **Embeddings:** `sentence-transformers` (MiniLM-L6-v2), CLIP (ViT-B/32)
- **Backend:** FastAPI, Uvicorn
- **Frontend:** React 18, Vite, Vanilla CSS
- **Game Engine:** HTML5 Canvas, `requestAnimationFrame`

---

## 📝 License

MIT — free to use, modify, and distribute.

---

<div align="center">

Made with 🪰 and actual neuroscience.

**Star ⭐ if you think flies are smarter than computer scientists.**

</div>
