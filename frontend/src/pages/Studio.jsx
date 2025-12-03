import React, { useEffect, useMemo, useState } from 'react'

const DEFAULT_BACKEND = 'http://127.0.0.1:8000'

export default function AppStudio(){
  // Backend + service state
  const [backend, setBackend] = useState(DEFAULT_BACKEND)
  const [health, setHealth] = useState(null)
  const [err, setErr] = useState('')

  // Local uploads
  const [fileA, setFileA] = useState(null)
  const [fileB, setFileB] = useState(null)
  const [seed, setSeed] = useState(0)

  // Fusion/style params
  const [method, setMethod] = useState('half')            // half | headbody | leftright | maskblend | offset | diag | graphcut | pyramid | parts3
  const [style, setStyle] = useState('illustrative')      // illustrative | realistic-soft | sketch
  const [harm, setHarm] = useState(true)
  const [harmAmount, setHarmAmount] = useState(0.35)      // 0..1
  const [featherPx, setFeatherPx] = useState(6)           // 0..20

  // Results
  const [baseImg, setBaseImg] = useState(null)
  const [styledImg, setStyledImg] = useState(null)
  const [loading, setLoading] = useState(false)

  // Export panel
  const [upscaler, setUpscaler] = useState('pxnn')        // pxnn | scale2x | lanczos_edge
  const [scale, setScale]       = useState(4)             // 2..8
  const [bg, setBg]             = useState('none')        // none | halo | sunset
  const [exportUrl, setExportUrl] = useState(null)        // PNG or GIF data URL
  const [exportKind, setExportKind] = useState('png')     // 'png' | 'gif'

  // Derived: a quick label for health
  const healthLabel = useMemo(()=>{
    if (health == null) return '—'
    if (health?.ok) return `OK (${health?.creatures ?? 0} sprites)`
    return 'Unreachable'
  }, [health])

  // Fetch creatures & health whenever backend changes
  useEffect(()=>{
    let alive = true
    setErr('')
    setHealth(null)

    // health
    fetch(`${backend}/health`)
      .then(r => r.json())
      .then(d => { if (alive) setHealth(d) })
      .catch(e => { if (alive) setErr(String(e)) })

    return () => { alive = false }
  }, [backend])

  // Run fusion+style using local uploads
  const runAll = async () => {
    setLoading(true); setErr(''); setBaseImg(null); setStyledImg(null); setExportUrl(null)
    try {
      if(!fileA || !fileB) throw new Error('Please upload Parent A and Parent B.')
      const fd = new FormData()
      fd.append('imageA', fileA)
      fd.append('imageB', fileB)
      fd.append('seed', String(seed))
      fd.append('method', method)
      fd.append('style', style)
      fd.append('harmonize', String(harm))
      fd.append('harm_amount', String(harmAmount))
      fd.append('feather_px', String(featherPx))
      const res = await fetch(`${backend}/style_upload`, { method:'POST', body: fd })
      if(!res.ok){
        let txt = await res.text()
        try { const j = JSON.parse(txt); txt = j.detail || txt } catch {}
        throw new Error(txt)
      }
      const data = await res.json()
      setBaseImg(data.base || null)
      setStyledImg(data.styled || null)
    } catch(e){ 
      setErr(String(e)) 
    } finally { 
      setLoading(false) 
    }
  }

  // --- Export helpers --------------------------------------------------------
  async function callExport(payload){
    const res = await fetch(`${backend}/export`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    })
    if(!res.ok){
      // try to surface JSON error text nicely
      let txt = await res.text()
      try { const j = JSON.parse(txt); txt = j.detail || txt } catch {}
      throw new Error(txt)
    }
    const data = await res.json()
    // { png: "data:image/png;base64,..."} or { gif: "data:image/gif;base64,..."}
    if (data.png) { setExportKind('png'); return data.png }
    if (data.gif) { setExportKind('gif'); return data.gif }
    throw new Error('Unexpected export payload.')
  }

  async function doExport(){
    setErr(''); setExportUrl(null)
    try{
      const url = await callExport({
        // export by hash is not available here; reuse last styled/base
        parents:null,
        method, style,
        harmonize: harm, harm_amount: Number(harmAmount),
        feather_px: Number(featherPx),
        upscaler,
        scale: Number(scale),
        background: bg
      })
      setExportUrl(url)
    }catch(e){ setErr(String(e)) }
  }

  async function doCard(){
    setErr(''); setExportUrl(null)
    try{
      const url = await callExport({
        parents:null,
        method, style,
        harmonize: harm, harm_amount: Number(harmAmount),
        feather_px: Number(featherPx),
        upscaler,
        scale: Number(scale),
        background: 'halo',
        card: true
      })
      setExportUrl(url)
      setExportKind('png')
    }catch(e){ setErr(String(e)) }
  }

  async function doGif(){
    setErr(''); setExportUrl(null)
    try{
      const url = await callExport({
        parents:null,
        method, style,
        harmonize: harm, harm_amount: Number(harmAmount),
        feather_px: Number(featherPx),
        upscaler: 'pxnn',
        scale: Math.max(2, Number(scale)),
        background: 'none',
        animate: true
      })
      setExportUrl(url)
      setExportKind('gif')
    }catch(e){ setErr(String(e)) }
  }

  function downloadDataUrl(dataUrl, filename){
    const a = document.createElement('a')
    a.href = dataUrl
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  const canDownload = !!exportUrl
  const dlName = `fusion_export.${exportKind === 'gif' ? 'gif' : 'png'}`

  // ----------------------------------------------------------------------------

  return (
    <div className="container">
      <h2>Implementation Studio</h2>

      <div className="appPanel">
        <div className="controls">

          {/* Backend URL + Health */}
          <div className="row">
            <div style={{flex:1}}>
              <label>Backend URL</label>
              <input value={backend} onChange={e=>setBackend(e.target.value)} />
              <small className="muted">Default: {DEFAULT_BACKEND}</small>
            </div>
            <div style={{minWidth:180, paddingTop:24}}>
              <span className="muted">Health:</span>{' '}
              <span style={{fontWeight:600, color: health?.ok ? 'var(--ok, #1a9f4b)' : '#b33'}}>
                {healthLabel}
              </span>
            </div>
          </div>

          {/* Parents */}
          <div className="row">
            <div style={{flex:1}}>
              <label>Parent A (upload)</label>
              <input type="file" accept="image/*" onChange={e=>setFileA(e.target.files?.[0]||null)} />
            </div>
            <div style={{flex:1}}>
              <label>Parent B (upload)</label>
              <input type="file" accept="image/*" onChange={e=>setFileB(e.target.files?.[0]||null)} />
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
                <option value="offset">Offset Split (seedable)</option>
                <option value="diag">Diagonal Blend</option>
                <option value="graphcut">Edge-aware Seam (DP)</option>
                <option value="pyramid">Multi-scale (Laplacian)</option>
                <option value="parts3">Parts 3 (head/torso/legs)</option>
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
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={harmAmount}
                onChange={e=>setHarmAmount(e.target.value)}
              />
              <small className="muted">Higher pulls colors toward a unified palette.</small>
            </div>
            <div style={{flex:1}}>
              <label>Feather (px)</label>
              <input
                type="number"
                min="0"
                max="20"
                step="1"
                value={featherPx}
                onChange={e=>setFeatherPx(e.target.value)}
              />
              <small className="muted">Seam softening for cut-based methods.</small>
            </div>
          </div>

          <button className="action" onClick={runAll} disabled={loading}>
            {loading ? 'Generating…' : 'Generate Fusion + Styled'}
          </button>
        </div>

        {/* Errors */}
        {err && <pre style={{color:'crimson', whiteSpace:'pre-wrap'}}>{err}</pre>}

        {/* Results */}
        <div className="panelGrid" style={{marginTop:16}}>
          <div className="card">
            <h3>Stage 1 — Pixel Fusion</h3>
            {baseImg ? <img className="pixel" src={baseImg} width={256} height={256} alt="base fusion"/> : <p className="muted">—</p>}
          </div>
          <div className="card">
            <h3>Stage 2 — Styled Output</h3>
            {styledImg ? <img className="pixel" src={styledImg} width={256} height={256} alt="styled output"/> : <p className="muted">—</p>}
          </div>
        </div>
      </div>
    </div>
  )
}
