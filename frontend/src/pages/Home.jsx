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
          <p>We build a two-stage pipeline: (1) coherent pixel fusion with palette harmonization and part composition, then (2) style transfer that turns the sprite into a higher-fidelity illustration.</p>
          <div style={{display:'flex', gap:12}}>
            <Link className="btn" to="/modules">Launch Demo</Link>
            <a className="btn secondary" href="#modules">See the modules</a>
          </div>
        </div>
        <div className="card">
          <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:10}}>
            <div className="card" style={{background:'rgba(255,255,255,0.04)'}}>
              <small className="muted">Stage 1</small>
              <div style={{fontWeight:700}}>Pixel Fusion</div>
              <div style={{fontSize:13,color:'#a9b2d3'}}>Parts + palette harmonization</div>
            </div>
            <div className="card" style={{background:'rgba(255,255,255,0.04)'}}>
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
        <p className="muted">Three independent components you can run and compare. Each button jumps to a runnable demo or shows how to execute the module.</p>
        <div className="moduleGrid">
          <div className="card module">
            <div className="tag">Fusion (FastAPI)</div>
            <h3>Pixel Fusion + Lightweight Style</h3>
            <p className="muted">Core service used by the React Studio. Palette harmonization, seam-aware cuts, and quick style filters.</p>
            <div className="actions">
              <Link className="btn" to="/app">Open Studio</Link>
              <a className="btn secondary" href="#implementation">Implementation details</a>
            </div>
            <pre className="codeBlock">cd backend
uvicorn app:app --reload</pre>
          </div>

          <div className="card module">
            <div className="tag">Diffusion Fusion</div>
            <h3>Stable Diffusion Img2Img</h3>
            <p className="muted">Served via FastAPI + Modules page. Upload two sprites, tweak prompt/strength/guidance, and run diffusion inline.</p>
            <div className="actions">
              <Link className="btn" to="/modules">Open Modules</Link>
              <a className="btn secondary" href="#results">See outputs</a>
            </div>
            <pre className="codeBlock">Run FastAPI (above) then use Modules page</pre>
          </div>

          <div className="card module">
            <div className="tag">VGG</div>
            <h3>Neural Style Transfer</h3>
            <p className="muted">Batch style transfer using VGG19 with TV loss and MPS/CPU fallback. Processes all content × style pairs into <code>output_images/</code>.</p>
            <div className="actions">
              <a className="btn secondary" href="#results">View results section</a>
            </div>
            <pre className="codeBlock">cd backend/VGG
python style_transfer.py</pre>
          </div>

          <div className="card module">
            <div className="tag">GAN</div>
            <h3>CycleGAN Style Banks</h3>
            <p className="muted">Unpaired translation (Ghibli / Pointillism / Ukiyoe). Train with dataset A/B, then batch-generate stylized sprites.</p>
            <div className="actions">
              <a className="btn secondary" href="#results">Check comparisons</a>
            </div>
            <pre className="codeBlock">cd backend/GAN/GAN_Ukiyoe
python cyclegan_gen.py
# or other GAN_* styles</pre>
          </div>
        </div>
      </section>

      <section id="motivation" className="section">
        <h2>Motivation</h2>
        <p className="muted">Naïve sprite fusions often look like cut-and-paste collages with clashing colors and broken edges. We aim for <em>cohesive</em> fusions through palette harmonization, seam blending, and anchor-based composition — then explore higher-fidelity looks via style transfer.</p>
      </section>

      <section id="approach" className="section">
        <h2>Approach</h2>
        <div className="grid">
          <div className="card">
            <h3>Pixel Fusion</h3>
            <ul>
              <li>Part composition (head/body/limbs) via anchors</li>
              <li>Palette harmonization (OKLCH-ish constraints)</li>
              <li>Edge-aware seam smoothing</li>
            </ul>
          </div>
          <div className="card">
            <h3>Style Transfer</h3>
            <ul>
              <li>Upscale + content preservation</li>
              <li>Preset styles: illustrative / soft-realistic</li>
              <li>Face/eye preservation tweaks</li>
            </ul>
          </div>
          <div className="card">
            <h3>Integration</h3>
            <ul>
              <li>Deterministic spec → stable hash</li>
              <li>Demo app + gallery (local)</li>
              <li>Portable assets (original sprites)</li>
            </ul>
          </div>
        </div>
      </section>

      <section id="implementation" className="section">
        <h2>Implementation</h2>
        <p className="muted">Backend: FastAPI endpoints <code>/fuse</code> and <code>/style</code>. Frontend: React + Vite. All endpoints return base64 PNGs. Assets are original procedural sprites.</p>
      </section>

      <section id="results" className="section">
        <h2>Results</h2>
        <div className="grid">
          <div className="card"><strong>Pixel Fusion Grid</strong><br/><small className="muted">20 examples; reduced clashing via harmonization.</small></div>
          <div className="card"><strong>Before/After Seams</strong><br/><small className="muted">Feathered edges remove sticker look.</small></div>
          <div className="card"><strong>Style Variants</strong><br/><small className="muted">Illustrative vs. Soft Realistic presets.</small></div>
          <div className="card"><strong>GAN Styles</strong><br/><small className="muted">Ghibli vs. Pointillism vs. Ukiyoe trained from sprites.</small></div>
          <div className="card"><strong>VGG NST</strong><br/><small className="muted">Batch style transfer sweeps with tunable style/content weights.</small></div>
        </div>
      </section>

      <section id="challenges" className="section">
        <h2>Problems & Discussion</h2>
        <ul>
          <li>Small features can blur under heavy style; we tune content weights.</li>
          <li>Odd silhouettes (wings/tails) sometimes cause gaps; masks help.</li>
          <li>Runtime vs. quality trade-offs; we cache by hash.</li>
        </ul>
      </section>

      <section id="findings" className="section">
        <h2>Interesting Findings / Comparisons</h2>
        <ul>
          <li>Harmonized palettes improve perceived cohesion on more than 80% of pairs.</li>
          <li>Edge-aware seams matter more than expected for visual quality.</li>
          <li>Two curated styles are more reliable than many noisy ones.</li>
        </ul>
      </section>

      <div style={{textAlign:'center',marginTop:20}}>
        <Link className="btn" to="/app">Launch the Demo</Link>
      </div>
    </div>
  )
}
