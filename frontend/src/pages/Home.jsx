import React from 'react'
import { Link } from 'react-router-dom'

function useGallery(backend) {
  const [items, setItems] = React.useState([]);
  React.useEffect(()=>{
    fetch(`${backend || 'http://127.0.0.1:8000'}/gallery?n=9`)
      .then(r=>r.json())
      .then(d=>setItems(d.items||[]))
      .catch(()=>{});
  },[backend]);
  return items;
}


export default function Home(){
  return (
    <div className="container">
      <section className="hero">
        <div>
          <div className="kicker">CS566 Project</div>
          <h1>Pokémon Fusion Art Generator</h1>
          <p className="muted" style={{marginBottom:6}}>Team: Tony Guo (hguo246) • Zhiyi Lai (zlai26) • Hua Zhou (hzhou397)</p>
          <p>Two-stage system: palette-coherent pixel fusion, then style transfer to lift sprites into richer art. All modules (Fusion, Diffusion, VGG, GAN) run from this React app.</p>
          <div style={{display:'flex', gap:12}}>
            <Link className="btn" to="/modules">Launch Demo</Link>
            <a className="btn secondary" href="#modules">Modules</a>
          </div>
          <div className="pillRow">
            <div className="pill">Fusion</div>
            <div className="pill">Diffusion</div>
            <div className="pill">VGG NST</div>
            <div className="pill">CycleGAN styles</div>
          </div>
        </div>
        <div className="card">
          <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:10}}>
            <div className="card" style={{background:'rgba(255,255,255,0.03)'}}>
              <small className="muted">Stage 1</small>
              <div style={{fontWeight:700}}>Pixel Fusion</div>
              <div style={{fontSize:13,color:'#a9b2d3'}}>Parts + palette harmonization</div>
            </div>
            <div className="card" style={{background:'rgba(255,255,255,0.03)'}}>
              <small className="muted">Stage 2</small>
              <div style={{fontWeight:700}}>Style Transfer</div>
              <div style={{fontSize:13,color:'#a9b2d3'}}>Illustrative or soft-realistic</div>
            </div>
          </div>
          <div style={{fontSize:13, color:'#a9b2d3',marginTop:8}}>Deterministic spec → hash for reproducible results.</div>
        </div>
      </section>

      <section id="modules" className="section">
        <h2>Modules & Demos</h2>
        <p className="muted">Three components you can run and compare. Use the Modules page for all demos.</p>
        <div className="moduleGrid">
          <div className="card module">
            <div className="tag">Fusion (FastAPI)</div>
            <h3>Pixel Fusion + Lightweight Style</h3>
            <p className="muted">Core service used by the React Studio. Palette harmonization, seam-aware cuts, and quick style filters.</p>
            <div className="actions">
              <Link className="btn" to="/app">Open Studio</Link>
              <a className="btn secondary" href="#implementation">Implementation details</a>
            </div>
          </div>

          <div className="card module">
            <div className="tag">Diffusion Fusion</div>
            <h3>Stable Diffusion Img2Img</h3>
            <p className="muted">Served via FastAPI + Modules page. Upload two sprites, tweak prompt/strength/guidance, and run diffusion inline.</p>
            <div className="actions">
              <Link className="btn" to="/modules">Open Modules</Link>
              <a className="btn secondary" href="#results">See outputs</a>
            </div>
          </div>

          <div className="card module">
            <div className="tag">VGG</div>
            <h3>Neural Style Transfer</h3>
            <p className="muted">Batch style transfer using VGG19 with TV loss and MPS/CPU fallback. Processes all content × style pairs into <code>output_images/</code>.</p>
            <div className="actions">
              <a className="btn secondary" href="#results">View results section</a>
            </div>
          </div>

          <div className="card module">
            <div className="tag">GAN</div>
            <h3>CycleGAN Style Banks</h3>
            <p className="muted">Unpaired translation (Ghibli / Pointillism / Ukiyoe). Train with dataset A/B, then batch-generate stylized sprites.</p>
            <div className="actions">
              <a className="btn secondary" href="#results">Check comparisons</a>
            </div>
          </div>
        </div>
      </section>

      <section className="section">
        <h2>Team</h2>
        <p className="muted">CS566 — Tony Guo (hguo246), Zhiyi Lai (zlai26), Hua Zhou (hzhou397)</p>
      </section>

      <section id="motivation" className="section">
        <h2>Motivation</h2>
        <p className="muted">Naïve fusion tools are collage-like: clashing palettes, broken seams, low fidelity. We pursue cohesive pixel fusions and then “lift” them into richer styles.</p>
      </section>

      <section id="approach" className="section">
        <h2>Approach</h2>
        <div className="grid">
          <div className="card">
            <h3>Problem & Goals</h3>
            <ul>
              <li>Fix aesthetic incoherence (palette + structure).</li>
              <li>Go beyond low-fi sprites via style transfer.</li>
              <li>Deliver reproducible, hashable outputs.</li>
            </ul>
          </div>
          <div className="card">
            <h3>Solution Plan</h3>
            <ul>
              <li>Part-aware fusion (head/body/limbs) + palette harmonization.</li>
              <li>Seam feathering, shading normalization.</li>
              <li>Style transfer: VGG NST, CycleGAN styles, diffusion img2img.</li>
            </ul>
          </div>
          <div className="card">
            <h3>Why it Matters</h3>
            <ul>
              <li>Gives artists coherent fusions and mod-friendly assets.</li>
              <li>Enables exploration of higher-fidelity variants.</li>
              <li>Teaches reproducible creative pipelines.</li>
            </ul>
          </div>
        </div>
      </section>

      <section id="implementation" className="section">
        <h2>Implementation</h2>
        <div className="grid">
          <div className="card">
            <h3>Fusion (FastAPI)</h3>
            <ul>
              <li>Endpoints: <code>/fuse</code>, <code>/style</code>, <code>/style_upload</code>, <code>/export</code>.</li>
              <li>Methods: half / headbody / leftright / maskblend / diag / graphcut / pyramid / parts3.</li>
              <li>Palette harmonization + seam feathering; upscalers + backgrounds.</li>
            </ul>
            <pre className="codeBlock">{`# style_upload (FastAPI)
@app.post("/style_upload")
async def style_upload(imageA: UploadFile, imageB: UploadFile, method: str="half"):
    a = Image.open(io.BytesIO(await imageA.read())).convert("RGBA")
    b = Image.open(io.BytesIO(await imageB.read())).convert("RGBA")
    fused  = _fuse_from_images(a, b, method, True, 0.35, 6)
    styled = stylize_filter(fused, "illustrative")
    return {"base": png_b64(fused), "styled": png_b64(styled)}`}</pre>
          </div>
          <div className="card">
            <h3>Diffusion (Img2Img)</h3>
            <ul>
              <li>Stable Diffusion img2img via <code>/diffusion/fuse</code>.</li>
              <li>Prompt, strength, guidance; uploads or samples.</li>
            </ul>
            <pre className="codeBlock">{`# React call (Modules page)
const fd = new FormData();
fd.append('imageA', fileA);
fd.append('imageB', fileB);
fd.append('prompt', prompt);
fd.append('strength', strength);
fd.append('guidance', guidance);
fetch(\`\${backend}/diffusion/fuse\`, { method:'POST', body: fd });`}</pre>
          </div>
          <div className="card">
            <h3>VGG NST & GAN</h3>
            <ul>
              <li>VGG19 batch NST: <code>/vgg/run</code>, results at <code>/vgg/results</code>.</li>
              <li>CycleGAN style banks (Ghibli / Pointillism / Ukiyoe): <code>/gan/run</code>, <code>/gan/results</code>.</li>
            </ul>
            <pre className="codeBlock">{`# VGG NST runner
python backend/VGG/style_transfer.py

# GAN generator (style folder)
python backend/GAN/GAN_Ukiyoe/cyclegan_gen.py`}</pre>
          </div>
          <div className="card">
            <h3>Frontend (React + Vite)</h3>
            <ul>
              <li>/modules: run Fusion, Diffusion, VGG, GAN.</li>
              <li>/app: Fusion Studio with local uploads.</li>
              <li>Bright, responsive layout.</li>
            </ul>
            <pre className="codeBlock">{`// Studio uploads -> /style_upload
const fd = new FormData();
fd.append('imageA', fileA);
fd.append('imageB', fileB);
fd.append('method', method);
fd.append('style', style);
fetch(\`\${backend}/style_upload\`, { method:'POST', body: fd });`}</pre>
          </div>
        </div>
      </section>

      <section id="results" className="section">
        <h2>Results & Comparisons</h2>
        <div className="grid">
          <div className="card"><strong>Pixel Fusion Grid</strong><br/><small className="muted">Palette harmonization vs. naïve cut/paste.</small></div>
          <div className="card"><strong>Seam Feathering</strong><br/><small className="muted">Before/after edge-aware blending.</small></div>
          <div className="card"><strong>Style Variants</strong><br/><small className="muted">Illustrative vs. soft-realistic presets.</small></div>
          <div className="card"><strong>GAN Styles</strong><br/><small className="muted">Ghibli / Pointillism / Ukiyoe translations.</small></div>
          <div className="card"><strong>VGG NST</strong><br/><small className="muted">Batch runs with tunable content/style weights.</small></div>
        </div>
      </section>

      <section id="challenges" className="section">
        <h2>Problems Encountered</h2>
        <ul>
          <li>Small features can blur under heavy style; tuned content weights and face/eye preservation.</li>
          <li>Odd silhouettes create gaps; masks and feathering help.</li>
          <li>Runtime vs. quality trade-offs; cached by hash and offered light/heavy presets.</li>
        </ul>
      </section>

      <section id="findings" className="section">
        <h2>Interesting Findings</h2>
        <ul>
          <li>Palette harmonization improves perceived cohesion on most pairs.</li>
          <li>Edge-aware seams matter more than expected for visual quality.</li>
          <li>A small curated style set outperforms many noisy ones.</li>
        </ul>
      </section>

      {/* Timeline removed per request */}
    </div>
  )
}
