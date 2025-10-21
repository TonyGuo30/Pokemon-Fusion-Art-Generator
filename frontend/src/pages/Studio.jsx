
import React, { useEffect, useState } from 'react'

const DEFAULT_BACKEND = 'http://127.0.0.1:8000'

export default function AppStudio(){
  const [backend, setBackend] = useState(DEFAULT_BACKEND)
  const [creatures, setCreatures] = useState([])
  const [a, setA] = useState('flamara')
  const [b, setB] = useState('aquaphin')
  const [seed, setSeed] = useState(0)

  // NEW: fusion/style params
  const [method, setMethod] = useState('half') // half | headbody | leftright | maskblend
  const [style, setStyle] = useState('illustrative') // illustrative | realistic-soft | sketch
  const [harm, setHarm] = useState(true)
  const [harmAmount, setHarmAmount] = useState(0.35) // 0..1
  const [featherPx, setFeatherPx] = useState(6) // 0..12

  const [baseImg, setBaseImg] = useState(null)
  const [styledImg, setStyledImg] = useState(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState('')

  useEffect(()=>{
    fetch(`${backend}/creatures`).then(r=>r.json()).then(setCreatures).catch(e=>setErr(String(e)))
  },[backend])

  const runAll = async () => {
    setLoading(true); setErr(''); setBaseImg(null); setStyledImg(null)
    try {
      const body = {
        parents:[a,b],
        seed:Number(seed),
        method,
        style,
        harmonize:harm,
        harm_amount: Number(harmAmount),
        feather_px: Number(featherPx)
      }
      const res = await fetch(`${backend}/style`, {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(body)
      })
      if(!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setBaseImg(data.base)
      setStyledImg(data.styled)
    } catch(e){ setErr(String(e)) } finally { setLoading(false) }
  }

  return (
    <div className="container">
      <h2>Implementation Demo</h2>
      <div className="appPanel">
        <div className="controls">

          {/* Backend URL */}
          <div className="row">
            <div style={{flex:1}}>
              <label>Backend URL</label>
              <input value={backend} onChange={e=>setBackend(e.target.value)} />
              <small className="muted">Default: http://127.0.0.1:8000</small>
            </div>
          </div>

          {/* Parents */}
          <div className="row">
            <div style={{flex:1}}>
              <label>Parent A</label>
              <select value={a} onChange={e=>setA(e.target.value)}>
                {creatures.map(c=> <option key={c.id} value={c.id}>{c.name} ({c.id})</option>)}
              </select>
            </div>
            <div style={{flex:1}}>
              <label>Parent B</label>
              <select value={b} onChange={e=>setB(e.target.value)}>
                {creatures.map(c=> <option key={c.id} value={c.id}>{c.name} ({c.id})</option>)}
              </select>
            </div>
          </div>

          {/* Method + Style */}
          <div className="row">
            <div style={{flex:1}}>
              <label>Fusion method</label>
              <select value={method} onChange={e=>setMethod(e.target.value)}>
                <option value="half">Half (top/bottom)</option>
                <option value="headbody">Head + Body</option>
                <option value="leftright">Left / Right</option>
                <option value="maskblend">Mask Blend (organic)</option>
              </select>
            </div>
            <div style={{flex:1}}>
              <label>Style</label>
              <select value={style} onChange={e=>setStyle(e.target.value)}>
                <option value="illustrative">Illustrative</option>
                <option value="realistic-soft">Realistic (soft)</option>
                <option value="sketch">Sketch</option>
              </select>
            </div>
          </div>

          {/* Seed + harmonization */}
          <div className="row">
            <div style={{flex:1}}>
              <label>Seed</label>
              <input type="number" value={seed} onChange={e=>setSeed(e.target.value)} />
            </div>
            <div style={{flex:1, display:'flex', gap:16, alignItems:'center', paddingTop:22}}>
              <label style={{display:'flex', gap:8, alignItems:'center'}}>
                <input type="checkbox" checked={harm} onChange={e=>setHarm(e.target.checked)} />
                Palette harmonization
              </label>
            </div>
          </div>

          {/* Harm amount + feather */}
          <div className="row">
            <div style={{flex:1}}>
              <label>Harmony strength (0–1)</label>
              <input type="number" min="0" max="1" step="0.05" value={harmAmount}
                     onChange={e=>setHarmAmount(e.target.value)} />
              <small className="muted">Higher pulls colors toward a unified palette.</small>
            </div>
            <div style={{flex:1}}>
              <label>Feather (px)</label>
              <input type="number" min="0" max="20" step="1" value={featherPx}
                     onChange={e=>setFeatherPx(e.target.value)} />
              <small className="muted">Seam softening for cut-based methods.</small>
            </div>
          </div>

          <button className="action" onClick={runAll} disabled={loading}>
            {loading ? 'Generating…' : 'Generate Fusion + Styled'}
          </button>
        </div>

        {err && <pre style={{color:'crimson', whiteSpace:'pre-wrap'}}>{err}</pre>}

        <div className="panelGrid" style={{marginTop:16}}>
          <div className="card">
            <h3>Stage 1 — Pixel Fusion</h3>
            {baseImg ? <img className="pixel" src={baseImg} width={256} height={256} /> : <p className="muted">—</p>}
          </div>
          <div className="card">
            <h3>Stage 2 — Styled Output</h3>
            {styledImg ? <img className="pixel" src={styledImg} width={256} height={256} /> : <p className="muted">—</p>}
          </div>
        </div>
      </div>
    </div>
  )
}
