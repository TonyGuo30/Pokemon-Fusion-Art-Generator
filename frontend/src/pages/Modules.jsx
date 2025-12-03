import React, { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

const DEFAULT_BACKEND = 'http://127.0.0.1:8000'

export default function Modules(){
  const [backend, setBackend] = useState(DEFAULT_BACKEND)

  // Diffusion (now via FastAPI) states
  const [prompt, setPrompt] = useState('fusion creature combining both Pokémon in anime style')
  const [strength, setStrength] = useState(0.75)
  const [guidance, setGuidance] = useState(7.5)
  const [imgA, setImgA] = useState(null)
  const [imgB, setImgB] = useState(null)
  const [diffusionOut, setDiffusionOut] = useState(null)
  const [diffusionMsg, setDiffusionMsg] = useState('')
  const [diffusionLoading, setDiffusionLoading] = useState(false)

  // Galleries
  const [vggItems, setVggItems] = useState([])
  const [ganItems, setGanItems] = useState([])
  const [vggMsg, setVggMsg] = useState('')
  const [ganMsg, setGanMsg] = useState('')
  const [ganStyle, setGanStyle] = useState('GAN_Ukiyoe')
  const [ganStyles, setGanStyles] = useState(['GAN_Ukiyoe'])
  const [runningVgg, setRunningVgg] = useState(false)
  const [runningGan, setRunningGan] = useState(false)

  const disableDiffusion = !imgA || !imgB || diffusionLoading

  async function runDiffusion(){
    setDiffusionMsg(''); setDiffusionOut(null); setDiffusionLoading(true)
    try{
      const fd = new FormData()
      fd.append('use_samples', 'false')
      fd.append('imageA', imgA)
      fd.append('imageB', imgB)
      fd.append('prompt', prompt)
      fd.append('strength', String(strength))
      fd.append('guidance', String(guidance))
      const res = await fetch(`${backend}/diffusion/fuse`, { method:'POST', body: fd })
      if(!res.ok) throw new Error(await res.text())
      const blob = await res.blob()
      setDiffusionOut(URL.createObjectURL(blob))
    }catch(e){ setDiffusionMsg(String(e)) }
    finally{ setDiffusionLoading(false) }
  }

  async function loadVgg(){
    try{
      const res = await fetch(`${backend}/vgg/results?n=12`)
      if(!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setVggItems(data.items || [])
    }catch(e){ setVggMsg(String(e)) }
  }
  async function loadGan(){
    try{
      const res = await fetch(`${backend}/gan/results?n=12`)
      if(!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setGanItems(data.items || [])
    }catch(e){ setGanMsg(String(e)) }
  }
  async function fetchGanStyles(){
    try{
      const res = await fetch(`${backend}/gan/styles`)
      if(!res.ok) return
      const data = await res.json()
      if(Array.isArray(data.styles) && data.styles.length>0){
        setGanStyles(data.styles)
        setGanStyle(data.styles[0])
      }
    }catch{}
  }
  async function runVgg(){
    setRunningVgg(true); setVggMsg('')
    try{
      const res = await fetch(`${backend}/vgg/run`, {method:'POST'})
      const data = await res.json()
      if(!res.ok) throw new Error(data.detail || JSON.stringify(data))
      setVggMsg(data.stdout || 'Done')
      loadVgg()
    }catch(e){ setVggMsg(String(e)) }
    finally{ setRunningVgg(false) }
  }
  async function runGan(){
    setRunningGan(true); setGanMsg('')
    try{
      const res = await fetch(`${backend}/gan/run?style=${encodeURIComponent(ganStyle)}`, {method:'POST'})
      const data = await res.json()
      if(!res.ok) throw new Error(data.detail || JSON.stringify(data))
      setGanMsg(data.stdout || 'Done')
      loadGan()
    }catch(e){ setGanMsg(String(e)) }
    finally{ setRunningGan(false) }
  }

  useEffect(()=>{
    fetchGanStyles()
    loadVgg()
    loadGan()
  }, [backend])

  const galleryNote = useMemo(()=>{
    return "Ensure backend is running and outputs exist in backend/VGG/output_images or backend/GAN/*/result."
  },[])

  return (
    <div className="container">
      <section className="section">
        <div className="kicker">Modules</div>
        <h1>Run Fusion, Diffusion, VGG NST, and GAN Styles</h1>
        <p className="muted">Simplified controls to exercise each module from one place.</p>
      </section>

      <section className="section card">
        <div className="row" style={{gap:12, flexWrap:'wrap'}}>
          <div style={{flex:1}}>
            <label>Backend URL</label>
            <input value={backend} onChange={e=>setBackend(e.target.value)} />
            <small className="muted">Default: {DEFAULT_BACKEND}</small>
          </div>
          <div style={{minWidth:200, display:'flex', alignItems:'flex-end'}}>
            <Link className="btn" to="/app">Open Fusion Studio</Link>
          </div>
        </div>
      </section>

      <section className="section card">
        <h3>Diffusion (Stable Diffusion Img2Img)</h3>
        <p className="muted">Upload two sprites, add a prompt, and run via FastAPI.</p>
        <div className="controls" style={{gridTemplateColumns:'repeat(auto-fit,minmax(220px,1fr))'}}>
          <div>
            <label>Prompt</label>
            <input value={prompt} onChange={e=>setPrompt(e.target.value)} />
          </div>
          <div>
            <label>Strength</label>
            <input type="number" step="0.05" min="0.3" max="1" value={strength} onChange={e=>setStrength(e.target.value)} />
          </div>
          <div>
            <label>Guidance</label>
            <input type="number" step="0.5" min="5" max="15" value={guidance} onChange={e=>setGuidance(e.target.value)} />
          </div>
          <div>
            <label>Image A (Base)</label>
            <input type="file" accept="image/*" onChange={e=>setImgA(e.target.files?.[0]||null)} />
          </div>
          <div>
            <label>Image B (Secondary)</label>
            <input type="file" accept="image/*" onChange={e=>setImgB(e.target.files?.[0]||null)} />
          </div>
        </div>
        <button className="action" onClick={runDiffusion} disabled={disableDiffusion}>
          {diffusionLoading ? 'Running diffusion…' : 'Run diffusion fusion'}
        </button>
        {diffusionMsg && <pre className="codeBlock" style={{marginTop:8, whiteSpace:'pre-wrap'}}>{diffusionMsg}</pre>}
        {diffusionOut && (
          <div style={{marginTop:12}}>
            <img src={diffusionOut} alt="diffusion result" className="pixel" style={{maxWidth:'100%'}}/>
          </div>
        )}
      </section>

      <section className="section card">
        <h3>VGG Neural Style Transfer</h3>
        <p className="muted">{galleryNote}</p>
        <div style={{display:'flex', gap:12, flexWrap:'wrap'}}>
          <button className="action" onClick={runVgg} disabled={runningVgg}>{runningVgg ? 'Running…' : 'Run batch'}</button>
          <button className="action" onClick={loadVgg}>Refresh results</button>
        </div>
        {vggMsg && <pre className="codeBlock" style={{marginTop:8, whiteSpace:'pre-wrap'}}>{vggMsg}</pre>}
        <div className="grid" style={{gridTemplateColumns:'repeat(auto-fit,minmax(180px,1fr))', marginTop:12}}>
          {vggItems.map(item=>(
            <div key={item.name} className="card" style={{padding:8}}>
              <small className="muted">{item.name}</small>
              <img src={item.image} alt={item.name} style={{width:'100%', borderRadius:8, marginTop:6}}/>
            </div>
          ))}
          {vggItems.length === 0 && <p className="muted">No results yet. Run the script, then refresh.</p>}
        </div>
      </section>

      <section className="section card">
        <h3>GAN Style Banks (CycleGAN)</h3>
        <p className="muted">{galleryNote}</p>
        <div style={{display:'flex', gap:12, flexWrap:'wrap', alignItems:'flex-end'}}>
          <div>
            <label>GAN Style Folder</label>
            <select value={ganStyle} onChange={e=>setGanStyle(e.target.value)}>
              {ganStyles.map(s=> <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <button className="action" onClick={runGan} disabled={runningGan}>{runningGan ? 'Running…' : 'Run GAN generator'}</button>
          <button className="action" onClick={loadGan}>Refresh results</button>
          <button className="action secondary" onClick={fetchGanStyles} style={{border:'1px solid rgba(255,255,255,0.12)', background:'transparent'}}>Reload styles</button>
        </div>
        {ganMsg && <pre className="codeBlock" style={{marginTop:8, whiteSpace:'pre-wrap'}}>{ganMsg}</pre>}
        <div className="grid" style={{gridTemplateColumns:'repeat(auto-fit,minmax(180px,1fr))', marginTop:12}}>
          {ganItems.map(item=>(
            <div key={item.name} className="card" style={{padding:8}}>
              <small className="muted">{item.name}</small>
              <img src={item.image} alt={item.name} style={{width:'100%', borderRadius:8, marginTop:6}}/>
            </div>
          ))}
          {ganItems.length === 0 && <p className="muted">No results yet. Generate outputs, then refresh.</p>}
        </div>
      </section>
    </div>
  )
}
