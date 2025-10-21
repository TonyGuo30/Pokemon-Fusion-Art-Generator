import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import Home from './pages/Home.jsx'
import AppStudio from './pages/Studio.jsx'
import './styles.css'

function Layout({children}){
  return (
    <div>
      <nav>
        <div className="brand">Pokémon Fusion Art</div>
        <div style={{display:'flex', gap:10}}>
          <Link className="btn secondary" to="/">Overview</Link>
          <Link className="btn" to="/app">Launch Demo</Link>
        </div>
      </nav>
      {children}
      <footer className="footer">© 2025 CS566 • Fusion Art Project</footer>
    </div>
  )
}

createRoot(document.getElementById('root')).render(
  <BrowserRouter>
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/app" element={<AppStudio />} />
      </Routes>
    </Layout>
  </BrowserRouter>
)
