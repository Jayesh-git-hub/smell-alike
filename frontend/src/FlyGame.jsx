/**
 * FlyGame — Authentic Drosophila Mushroom Body Game
 * ───────────────────────────────────────────────────────────────────────────
 * What makes this DIFFERENT from Chrome Dino:
 *
 * The fly has a REAL associative learning circuit based on the Drosophila
 * mushroom body connectome. Each obstacle type gets a UNIQUE sparse KC
 * fingerprint (like an odor code). When the fly crashes into a specific
 * obstacle, the dopamine system strengthens the KC→MBON weight for that
 * obstacle's fingerprint — so next time it sees that same obstacle, the
 * MBON fires strongly and auto-triggers a jump BEFORE the user even reacts.
 *
 * Player controls the fly manually (Space/tap). The fly's neural circuit
 * runs in parallel and can "assist" by auto-jumping if it has learned that
 * obstacle is dangerous. The more crashes, the smarter the assist.
 *
 * ───── Real Biology ─────
 * Projection Neurons (PNs) → Kenyon Cells (KCs) in Mushroom Body (MB)
 *   • Each KC receives input from exactly 6 random PNs out of 50
 *     (Caron et al. 2013, Nature — "Randomness and specificity in MB wiring")
 *   • We scale: each KC receives input from exactly 3 out of 12 PNs
 *   • Sparse WTA: only top 5% of KCs stay active (APL neuron inhibition)
 *
 * KC → MBON (Mushroom Body Output Neuron)
 *   • MBON-γ1pedc>αβ mediates learned avoidance
 *   • Dopaminergic PAM neurons modulate KC→MBON weights
 *   • Crash = DAL/PPL1 dopamine (punishment) → suppress current KC pattern
 *   • Clear = PAM dopamine (reward) → reinforce current KC pattern
 *
 * Sources:
 *   Aso et al. (2014) eLife — "Mushroom body output neurons encode valence"
 *   Caron et al. (2013) Nature — "Random wiring in the fly MB"
 *   Li et al. (2020) — "The connectome of a larval Drosophila MB"
 *   Dasgupta, Stevens & Navlakha (2017) Science — FlyHash algorithm
 */

import { useState, useEffect, useRef, useCallback } from "react"

/* ─── Connectome Constants (scaled from real biology) ─── */
const N_PN    = 12        // Projection Neurons (input layer)
const N_KC    = 40        // Kenyon Cells in MB (real fly has ~2000)
const KC_SYNAPSES = 3     // Each KC receives from exactly 3 PNs (Caron 2013: 6/50)
const WTA_FRAC = 0.05     // Top 5% KC survive APL inhibition
const WTA_K    = Math.max(1, Math.round(N_KC * WTA_FRAC)) // = 2 KCs fire

const LR_POS  = 0.35    // PAM dopamine (reward) learning rate
const LR_NEG  = 0.55    // PPL1 dopamine (punishment) learning rate
const WEIGHT_DECAY = 0.999

const ASSIST_THRESHOLD = 0.68  // MBON above this → auto-jump assist fires

/* ─── Game Physics ─── */
const W = 900, H = 300
const GROUND  = H - 48
const FLY_X   = 90
const GRAVITY  = 0.57
const JUMP_VY  = -12.5
const FLY_W   = 30, FLY_H = 24
const OB_W    = 26

/* ─── Build sparse connectome (fixed wiring, like biology) ─── */
function buildConnectome() {
  // Each KC connects to exactly KC_SYNAPSES random PNs
  return Array.from({ length: N_KC }, (_, k) => {
    const row = new Array(N_PN).fill(0)
    // Sample KC_SYNAPSES unique PNs without replacement
    const pnIndices = []
    while (pnIndices.length < KC_SYNAPSES) {
      const idx = Math.floor(Math.random() * N_PN)
      if (!pnIndices.includes(idx)) pnIndices.push(idx)
    }
    pnIndices.forEach(i => { row[i] = Math.random() > 0.5 ? 1 : -1 })
    return row
  })
}

/**
 * Encode an obstacle into a PN input vector.
 * Each obstacle gets a DETERMINISTIC hash-based PN pattern
 * (like how different odors activate different olfactory receptors).
 */
function obstacleToPN(obstacleText) {
  // Simple deterministic hash of the text → PN activation pattern
  let hash = 0
  for (let i = 0; i < obstacleText.length; i++) {
    hash = (hash * 31 + obstacleText.charCodeAt(i)) >>> 0
  }
  // Generate N_PN values in [0,1] from the hash, sparse (only ~40% active)
  return Array.from({ length: N_PN }, (_, i) => {
    const h = ((hash * (i + 1) * 2654435761) >>> 0) / 0xFFFFFFFF
    return h > 0.6 ? h : 0  // sparse: only ~40% of PNs fire per odor
  })
}

/**
 * KC layer forward pass with WTA inhibition (APL neuron)
 */
function kcForward(pn, connW) {
  const acts = connW.map(row => row.reduce((s, w, i) => s + w * pn[i], 0))
  const sorted = [...acts].sort((a, b) => b - a)
  const thresh = sorted[WTA_K - 1] ?? 0
  return acts.map(a => (a >= thresh && a > 0 ? a : 0))
}

/** MBON: sigmoid output = learned avoidance strength */
function mbonForward(kc, wMBON) {
  const z = kc.reduce((s, a, i) => s + a * wMBON[i], 0)
  return 1 / (1 + Math.exp(-z * 3))  // sharper sigmoid
}

/** Dopamine weight update */
function updateWeights(wMBON, kc, reward) {
  const lr = reward > 0 ? LR_POS : LR_NEG
  return wMBON.map((w, i) => {
    if (kc[i] === 0) return w * WEIGHT_DECAY
    return w * WEIGHT_DECAY + lr * reward * kc[i]
  })
}

/** Jaccard similarity between two KC patterns (how "similar" two obstacles look) */
function kcSimilarity(kc1, kc2) {
  const a = new Set(kc1.map((v, i) => v > 0 ? i : -1).filter(i => i >= 0))
  const b = new Set(kc2.map((v, i) => v > 0 ? i : -1).filter(i => i >= 0))
  const inter = [...a].filter(x => b.has(x)).length
  const union = new Set([...a, ...b]).size
  return union === 0 ? 0 : inter / union
}

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

/* ═══════════════════════════════════════════════════════════════════════════ */

export default function FlyGame({ onBack }) {
  const [view,      setView]      = useState("config")
  const [obstacles, setObstacles] = useState(["🌵", "🕷️", "🦗", "📰"])
  const [newOb,     setNewOb]     = useState("")
  const [stats,     setStats]     = useState({
    score: 0, hi: 0, mbonProb: 0, activeKC: 0,
    speed: 4.5, lastOb: "", assistCount: 0, learnedObs: {}
  })
  const [deathInfo, setDeathInfo] = useState(null)
  const [showNeuro, setShowNeuro] = useState(true)

  const canvasRef = useRef(null)
  const gRef      = useRef(null)
  const nnRef     = useRef(null)
  const assistRef = useRef(0) // total assists given

  /* Build connectome once per session */
  if (!nnRef.current) {
    nnRef.current = {
      connW:   buildConnectome(),   // FIXED sparse wiring (biological constraint)
      wMBON:   new Array(N_KC).fill(0),  // MBON weights start at 0
      hi:      0,
      assistCount: 0,
      // Cache: obstacle text → KC fingerprint (computed on first encounter)
      kcCache: {},
      // Track which obstacles the fly has learned (crashes into)
      learned: {},
    }
  }

  /* Get or compute KC fingerprint for an obstacle */
  function getKC(nn, obText) {
    if (!nn.kcCache[obText]) {
      const pn = obstacleToPN(obText)
      nn.kcCache[obText] = kcForward(pn, nn.connW)
    }
    return nn.kcCache[obText]
  }

  /* ── Jump (user input) ── */
  const tryJump = useCallback((source = "user") => {
    const g = gRef.current
    if (!g || !g.alive || g.jumping) return false
    g.flyVy   = JUMP_VY
    g.jumping = true
    g.jumpSource = source
    return true
  }, [])

  /* ── Start game ── */
  const startGame = useCallback(() => {
    gRef.current = {
      flyY: GROUND - FLY_H,
      flyVy: 0,
      jumping: false,
      jumpSource: null,
      obs: [],
      speed: 4.5,
      frame: 0,
      score: 0,
      alive: true,
      assistFlash: 0,  // frames to show assist glow
    }
    setDeathInfo(null)
    assistRef.current = 0
    setView("playing")
  }, [])

  /* ── Keyboard ── */
  useEffect(() => {
    if (view !== "playing") return
    const fn = (e) => {
      if (e.code === "Space" || e.code === "ArrowUp") { e.preventDefault(); tryJump("user") }
    }
    window.addEventListener("keydown", fn)
    return () => window.removeEventListener("keydown", fn)
  }, [view, tryJump])

  /* ── Draw ── */
  const draw = useCallback((ctx, g) => {
    const nn = nnRef.current

    /* Sky */
    const sky = ctx.createLinearGradient(0, 0, 0, H)
    sky.addColorStop(0, "#030710"); sky.addColorStop(1, "#0b1022")
    ctx.fillStyle = sky; ctx.fillRect(0, 0, W, H)

    /* Stars */
    ctx.fillStyle = "rgba(255,255,255,0.16)"
    for (let i = 0; i < 50; i++) {
      ctx.fillRect((i * 137 + 41) % W, (i * 97 + 13) % (GROUND - 10), 1.2, 1.2)
    }

    /* Ground */
    ctx.strokeStyle = "rgba(0,229,204,0.28)"; ctx.lineWidth = 1.5
    ctx.beginPath(); ctx.moveTo(0, GROUND); ctx.lineTo(W, GROUND); ctx.stroke()
    const gGlow = ctx.createLinearGradient(0, GROUND, 0, H)
    gGlow.addColorStop(0, "rgba(0,229,204,0.06)"); gGlow.addColorStop(1, "transparent")
    ctx.fillStyle = gGlow; ctx.fillRect(0, GROUND, W, H - GROUND)

    /* Neural impulse lines (showNeuro) */
    if (showNeuro && nn.lastKC) {
      nn.lastKC.forEach((a, i) => {
        if (a > 0) {
          const hue = 160 + i * 9
          ctx.save()
          ctx.strokeStyle = `hsla(${hue}, 85%, 62%, 0.3)`
          ctx.lineWidth = 1.5
          ctx.setLineDash([4, 7])
          ctx.beginPath()
          ctx.moveTo(FLY_X + FLY_W / 2, g.flyY + FLY_H / 2)
          ctx.lineTo(20 + (i / N_KC) * (W - 40), GROUND + 24)
          ctx.stroke()
          ctx.restore()
        }
      })
    }

    /* Obstacles — colour-coded by how dangerous the fly thinks they are */
    g.obs.forEach(ob => {
      const kc   = getKC(nn, ob.text)
      const prob = mbonForward(kc, nn.wMBON)
      // Danger glow: red if high MBON, neutral otherwise
      if (prob > 0.4) {
        ctx.save()
        const grd = ctx.createRadialGradient(
          ob.x + ob.w / 2, GROUND - ob.h / 2, 2,
          ob.x + ob.w / 2, GROUND - ob.h / 2, ob.h * 1.4
        )
        grd.addColorStop(0, `rgba(255,80,80,${prob * 0.35})`)
        grd.addColorStop(1, "transparent")
        ctx.fillStyle = grd
        ctx.fillRect(ob.x - ob.h, GROUND - ob.h * 2, ob.w + ob.h * 2, ob.h * 2.5)
        ctx.restore()
      }
      ctx.font = `${ob.h}px serif`; ctx.textAlign = "center"; ctx.fillStyle = "#fff"
      ctx.fillText(ob.text, ob.x + ob.w / 2, GROUND - 2)
    })

    /* Fly — with assist glow */
    ctx.save()
    if (g.assistFlash > 0) {
      // Green glow: neural assist is firing
      ctx.shadowColor = "rgba(46,213,115,0.9)"
      ctx.shadowBlur  = 18 + g.assistFlash * 0.5
    }
    if (g.flyVy < -4) {
      ctx.translate(FLY_X + FLY_W / 2, g.flyY + FLY_H / 2)
      ctx.rotate(-0.3)
      ctx.translate(-(FLY_X + FLY_W / 2), -(g.flyY + FLY_H / 2))
    }
    ctx.font = `${FLY_H}px serif`; ctx.textAlign = "center"
    ctx.fillText("🪰", FLY_X + FLY_W / 2, g.flyY + FLY_H)
    ctx.restore()
    if (g.assistFlash > 0) g.assistFlash--

    /* Speed lines */
    if (g.speed > 7) {
      ctx.strokeStyle = `rgba(100,120,255,${Math.min((g.speed - 7) * 0.04, 0.22)})`
      ctx.lineWidth = 1
      for (let i = 0; i < 3; i++) {
        const ly = g.flyY + 5 + i * 8
        ctx.beginPath(); ctx.moveTo(FLY_X - 8 - i * 9, ly); ctx.lineTo(FLY_X - 22 - i * 9, ly); ctx.stroke()
      }
    }

    /* HUD */
    ctx.textAlign = "left"; ctx.font = "bold 13px 'JetBrains Mono', monospace"
    ctx.fillStyle = "rgba(0,229,204,0.85)";  ctx.fillText(`SCORE: ${g.score}`, 12, 24)
    ctx.fillStyle = "rgba(255,159,67,0.75)"; ctx.fillText(`BEST: ${nn.hi}`,   12, 42)
    ctx.fillStyle = "rgba(46,213,115,0.7)";  ctx.fillText(`ASSISTS: ${nn.assistCount}`, 12, 60)

    /* MBON danger bar — per closest obstacle */
    const closest = g.obs.find(o => o.x + o.w > FLY_X)
    if (closest) {
      const kc   = getKC(nn, closest.text)
      const prob = mbonForward(kc, nn.wMBON)
      const bx = W - 130, by = 14, bw = 116, bh = 8
      // Background
      ctx.fillStyle = "rgba(255,255,255,0.05)"
      ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, 4); ctx.fill()
      // Fill colour: green→yellow→red based on danger
      const r = Math.round(prob * 255), gr = Math.round((1 - prob) * 200)
      ctx.fillStyle = `rgb(${r},${gr},60)`
      ctx.beginPath(); ctx.roundRect(bx, by, bw * prob, bh, 4); ctx.fill()
      // Label
      ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.font = "10px 'JetBrains Mono', monospace"
      ctx.textAlign = "right"
      ctx.fillText(`DANGER ${(prob * 100).toFixed(0)}%`, W - 14, by - 3)
      ctx.textAlign = "left"
      // Assist warning pulse
      if (prob > ASSIST_THRESHOLD && !g.jumping) {
        ctx.fillStyle = `rgba(46,213,115,${0.5 + Math.sin(g.frame * 0.3) * 0.3})`
        ctx.font = "bold 12px 'Inter', sans-serif"; ctx.textAlign = "center"
        ctx.fillText("⚡ NEURAL ASSIST", W / 2, 22)
        ctx.textAlign = "left"
      }
    }

    /* Hint text */
    if (g.frame < 160) {
      ctx.fillStyle = `rgba(255,255,255,${Math.max(0, 0.7 - g.frame * 0.0044)})`
      ctx.font = "bold 13px 'Inter', sans-serif"; ctx.textAlign = "center"
      ctx.fillText("SPACE / TAP to jump — fly's brain assists when it recognises danger", W / 2, GROUND - 16)
      ctx.textAlign = "left"
    }
  }, [showNeuro])

  /* ── Game loop ── */
  useEffect(() => {
    if (view !== "playing") return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    let raf

    const loop = () => {
      const g  = gRef.current
      const nn = nnRef.current
      if (!g.alive) {
        nn.hi = Math.max(nn.hi, g.score)
        setDeathInfo({ score: g.score, hi: nn.hi, assists: nn.assistCount, learned: { ...nn.learned } })
        setStats(s => ({ ...s, score: g.score, hi: nn.hi, assistCount: nn.assistCount }))
        setView("gameover")
        return
      }

      g.frame++

      /* Spawn obstacle */
      const minGap = clamp(310 - g.speed * 14, 155, 310)
      const lastX  = g.obs.length ? g.obs[g.obs.length - 1].x : -999
      if (W - lastX > minGap && g.obs.length < 3) {
        const txt = obstacles[Math.floor(Math.random() * obstacles.length)] || "🌵"
        g.obs.push({ x: W + 10, text: txt, w: OB_W, h: 26 + Math.floor(Math.random() * 16), passed: false })
      }

      /* ── Neural pass: look at the CLOSEST obstacle ── */
      const closest = g.obs.find(o => o.x + o.w > FLY_X - 4)
      let kc = new Array(N_KC).fill(0)
      let prob = 0

      if (closest) {
        kc   = getKC(nn, closest.text)
        prob = mbonForward(kc, nn.wMBON)
        nn.lastKC   = kc
        nn.lastProb = prob

        /* ── NEURAL ASSIST: auto-jump if MBON danger above threshold ──
         * This is what makes it different from Dino:
         * The fly's brain recognises a previously-crashed obstacle
         * and fires the jump reflex automatically.
         */
        const dist = closest.x - FLY_X
        const REACT_DIST = 90 + g.speed * 6  // lookahead based on speed

        if (prob > ASSIST_THRESHOLD && dist < REACT_DIST && dist > 0) {
          const jumped = tryJump("neural")
          if (jumped) {
            g.assistFlash = 30
            nn.assistCount++
            assistRef.current = nn.assistCount
          }
        }
      }

      /* Physics */
      g.flyY  += g.flyVy
      g.flyVy += GRAVITY
      if (g.flyY >= GROUND - FLY_H) {
        g.flyY    = GROUND - FLY_H
        g.flyVy   = 0
        g.jumping = false
        g.jumpSource = null
      }

      /* Move obstacles */
      for (let i = g.obs.length - 1; i >= 0; i--) {
        g.obs[i].x -= g.speed
        if (!g.obs[i].passed && g.obs[i].x + g.obs[i].w < FLY_X) {
          g.obs[i].passed = true
          g.score++
          g.speed = Math.min(g.speed + 0.08, 18)
          // ── PAM dopamine: reward for clearing this obstacle ──
          const clearedKC = getKC(nn, g.obs[i].text)
          nn.wMBON = updateWeights(nn.wMBON, clearedKC, -0.1) // slight suppression (cleared = not dangerous)
          nn.learned[g.obs[i].text] = (nn.learned[g.obs[i].text] || 0)
        }
        if (g.obs[i].x + g.obs[i].w < 0) g.obs.splice(i, 1)
      }

      /* Collision */
      let crashed = false
      for (const ob of g.obs) {
        if (ob.x < FLY_X + FLY_W - 8 && ob.x + ob.w > FLY_X + 8 &&
            g.flyY + FLY_H - 7 > GROUND - ob.h) {
          // ── PPL1 dopamine: punishment for crashing into this obstacle ──
          const crashedKC = getKC(nn, ob.text)
          nn.wMBON = updateWeights(nn.wMBON, crashedKC, 1.0) // strong potentiation — LEARN THIS
          nn.learned[ob.text] = (nn.learned[ob.text] || 0) + 1
          g.alive = false
          crashed = true
          break
        }
      }

      /* Stats update */
      if (g.frame % 8 === 0) {
        setStats({
          score:    g.score,
          hi:       nn.hi,
          mbonProb: +(prob * 100).toFixed(1),
          activeKC: kc.filter(a => a > 0).length,
          speed:    +g.speed.toFixed(1),
          lastOb:   closest?.text || "",
          assistCount: nn.assistCount,
          learnedObs: { ...nn.learned },
        })
      }

      draw(ctx, g)
      raf = requestAnimationFrame(loop)
    }

    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [view, obstacles, draw, tryJump])

  const addOb    = () => { const v = newOb.trim(); if (!v || obstacles.includes(v)) return; setObstacles(p => [...p, v]); setNewOb("") }
  const removeOb = (o) => setObstacles(p => p.filter(x => x !== o))

  return (
    <div className="game-wrapper">
      {/* Top bar */}
      <div className="game-topbar">
        <button className="back-btn" onClick={onBack}>← Menu</button>
        <h2>🪰 Fruitfly Speedrun — Connectome AI</h2>
        <div className="game-score-panel">
          <div className="game-score-item"><div className="game-score-value">{stats.score}</div><div className="game-score-label">Score</div></div>
          <div className="game-score-item"><div className="game-score-value" style={{color:"var(--accent-orange)"}}>{stats.hi}</div><div className="game-score-label">Best</div></div>
          <div className="game-score-item"><div className="game-score-value" style={{color:"var(--accent-green)"}}>{stats.assistCount}</div><div className="game-score-label">Assists</div></div>
        </div>
      </div>

      {/* ── CONFIG ── */}
      {view === "config" && (
        <div className="game-config">
          <h3>🧬 Set up obstacles</h3>
          <p className="desc">
            Add the obstacles. <strong>Space / tap</strong> to jump.
            The fly's brain learns which obstacles are dangerous and auto-jumps to help you.
            The more you crash into something, the stronger the neural reflex becomes.
          </p>
          <div className="obstacle-chips">
            {obstacles.map(ob => (
              <span key={ob} className="obstacle-chip" onClick={() => removeOb(ob)} title="Click to remove">{ob} ✕</span>
            ))}
          </div>
          <div className="obstacle-input-row">
            <input type="text" placeholder="e.g. 🕸️ or 'newspaper'"
              value={newOb} onChange={e => setNewOb(e.target.value)}
              onKeyDown={e => e.key === "Enter" && addOb()} />
            <button className="add-obstacle-btn" onClick={addOb}>+ Add</button>
          </div>

          <div className="neuro-info">
            <strong>🧠 How the Drosophila mushroom body works here:</strong><br/><br/>
            Each obstacle type activates a <strong>unique sparse pattern</strong> across {N_KC} Kenyon Cells
            (like how different odors activate different olfactory receptor patterns).
            Each KC connects to exactly <strong>3 of {N_PN} Projection Neurons</strong> (Caron et al. 2013 — 6 of ~50 PNs in real fly).<br/><br/>
            When you crash into an obstacle, <strong>PPL1 dopamine neurons</strong> fire,
            strengthening that obstacle's KC→MBON weight. Next time the fly sees the same obstacle,
            the MBON output rises above the danger threshold ({(ASSIST_THRESHOLD * 100).toFixed(0)}%) and
            <strong> auto-triggers a jump</strong> — exactly how Drosophila learns to avoid dangerous odors.<br/><br/>
            The <strong>red glow</strong> around an obstacle shows how dangerous the fly currently thinks it is.
            The <strong>green glow</strong> on the fly means the neural assist just fired.
          </div>

          <button className="start-game-btn" onClick={startGame}>
            🚀 Start — crash to teach the fly!
          </button>
        </div>
      )}

      {/* ── PLAYING ── */}
      {view === "playing" && (
        <>
          <div className="game-canvas-wrap" onClick={() => tryJump("user")}>
            <canvas ref={canvasRef} width={W} height={H} className="game-canvas" />
          </div>

          <div className="neural-panel">
            <div className="neural-card">
              <div className="neural-label">Danger (MBON) for {stats.lastOb || "—"}</div>
              <div className="neural-value" style={{
                color: stats.mbonProb > 68 ? "var(--accent-red)" :
                       stats.mbonProb > 40 ? "var(--accent-orange)" : "var(--accent-teal)"
              }}>{stats.mbonProb}%</div>
            </div>
            <div className="neural-card">
              <div className="neural-label">Active KCs / {N_KC}</div>
              <div className="neural-value" style={{color:"var(--accent-purple)"}}>{stats.activeKC}</div>
            </div>
            <div className="neural-card">
              <div className="neural-label">Speed</div>
              <div className="neural-value">{stats.speed}×</div>
            </div>
          </div>

          <div style={{display:"flex", justifyContent:"space-between", alignItems:"center", marginTop:"0.75rem", padding:"0 0.25rem"}}>
            <p style={{color:"var(--text-dim)",fontSize:"0.8rem"}}>
              🎮 <strong>Space / ↑</strong> or tap &nbsp;·&nbsp;
              ⚡ Green glow = neural assist jumped
            </p>
            <label style={{display:"flex",alignItems:"center",gap:"6px",color:"var(--text-dim)",fontSize:"0.78rem",cursor:"pointer"}}>
              <input type="checkbox" checked={showNeuro} onChange={e => setShowNeuro(e.target.checked)} />
              Neural lines
            </label>
          </div>

          {/* Learned obstacles display */}
          {Object.keys(stats.learnedObs).length > 0 && (
            <div style={{marginTop:"0.75rem",padding:"10px 14px",background:"var(--bg-card)",border:"1px solid var(--border-subtle)",borderRadius:"var(--radius-md)"}}>
              <div style={{fontSize:"0.72rem",color:"var(--text-dim)",textTransform:"uppercase",letterSpacing:"0.05em",marginBottom:"6px"}}>
                🧠 Fly's learned crash memory
              </div>
              <div style={{display:"flex",flexWrap:"wrap",gap:"6px"}}>
                {Object.entries(stats.learnedObs).map(([ob, crashes]) => (
                  <span key={ob} style={{
                    padding:"3px 10px", borderRadius:"100px",
                    background:`rgba(255,80,80,${Math.min(0.15 + crashes * 0.12, 0.5)})`,
                    border:`1px solid rgba(255,80,80,${Math.min(0.2 + crashes * 0.15, 0.7)})`,
                    color:"var(--accent-red)", fontSize:"0.8rem"
                  }}>
                    {ob} × {crashes} crash{crashes > 1 ? "es" : ""}
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* ── GAME OVER ── */}
      {view === "gameover" && deathInfo && (
        <div className="game-over-panel">
          <h3>💥 Crashed!</h3>
          <p style={{color:"var(--text-secondary)",fontSize:"0.9rem",marginBottom:"1rem"}}>
            The fly's neural weights have been updated. It will recognise that obstacle next run.
          </p>
          <div className="game-over-stats">
            <div className="stat-card"><div className="stat-value">{deathInfo.score}</div><div className="stat-label">Score</div></div>
            <div className="stat-card">
              <div className="stat-value" style={{color:"var(--accent-green)"}}>{deathInfo.assists}</div>
              <div className="stat-label">Neural assists</div>
            </div>
          </div>
          {Object.keys(deathInfo.learned).length > 0 && (
            <div style={{marginBottom:"1rem",textAlign:"left",padding:"10px 14px",background:"rgba(255,80,80,0.06)",border:"1px solid rgba(255,80,80,0.2)",borderRadius:"var(--radius-md)"}}>
              <div style={{fontSize:"0.72rem",color:"var(--text-dim)",marginBottom:"5px",textTransform:"uppercase",letterSpacing:"0.05em"}}>Crash memory (stronger next run)</div>
              {Object.entries(deathInfo.learned).map(([ob, n]) => (
                <span key={ob} style={{marginRight:"8px",fontSize:"0.88rem",color:"var(--accent-red)"}}>{ob} ×{n}</span>
              ))}
            </div>
          )}
          <button className="restart-btn" onClick={startGame}>▶ Play Again (smarter fly)</button>
          <button className="back-btn" style={{marginTop:"0.75rem",width:"100%",textAlign:"center"}} onClick={() => setView("config")}>⚙ Change Obstacles</button>
        </div>
      )}
    </div>
  )
}