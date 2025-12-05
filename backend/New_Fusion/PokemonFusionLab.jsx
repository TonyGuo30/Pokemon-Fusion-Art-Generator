import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Shuffle, Download, Palette, Layers, Move, Eraser, Check, Hash, ArrowRightLeft } from 'lucide-react';

// --- COLOR UTILITIES ---

// Convert RGB to HSL
const rgbToHsl = (r, g, b) => {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s, l = (max + min) / 2;

  if (max === min) {
    h = s = 0; // achromatic
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }
    h /= 6;
  }
  return [h, s, l];
};

// Convert HSL to RGB
const hslToRgb = (h, s, l) => {
  let r, g, b;
  if (s === 0) {
    r = g = b = l; // achromatic
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
};

// Simple Euclidean color distance
const colorDistance = (rgb1, rgb2) => {
  return Math.sqrt(
    Math.pow(rgb1[0] - rgb2[0], 2) +
    Math.pow(rgb1[1] - rgb2[1], 2) +
    Math.pow(rgb1[2] - rgb2[2], 2)
  );
};

// --- APP COMPONENT ---

export default function PokemonFusionLab() {
  // State
  const [headId, setHeadId] = useState(1); // Bulbasaur
  const [bodyId, setBodyId] = useState(4); // Charmander
  const [headImage, setHeadImage] = useState(null);
  const [bodyImage, setBodyImage] = useState(null);
  
  // Fusion Parameters
  const [scale, setScale] = useState(0.8);
  const [offsetX, setOffsetX] = useState(0);
  const [offsetY, setOffsetY] = useState(-20);
  const [flipHead, setFlipHead] = useState(false);
  const [usePaletteSwap, setUsePaletteSwap] = useState(true);
  const [feathering, setFeathering] = useState(2);
  const [contrastBoost, setContrastBoost] = useState(1.1);
  
  // System
  const canvasRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [loading, setLoading] = useState(false);
  const [fusionHash, setFusionHash] = useState('');

  // 1. Fetch Images
  const fetchSprite = (id) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "Anonymous";
      // Using raw github pokeapi sprites for ease of access without CORS issues on some proxies
      img.src = `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/${id}.png`;
      img.onload = () => resolve(img);
      img.onerror = reject;
    });
  };

  const loadPokemon = async () => {
    setLoading(true);
    try {
      const [hImg, bImg] = await Promise.all([
        fetchSprite(headId),
        fetchSprite(bodyId)
      ]);
      setHeadImage(hImg);
      setBodyImage(bImg);
    } catch (e) {
      console.error("Failed to load sprites", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPokemon();
  }, [headId, bodyId]);

  // 2. Generate Hash (Determinism)
  useEffect(() => {
    const data = `${headId}-${bodyId}-${scale.toFixed(2)}-${offsetX}-${offsetY}-${usePaletteSwap ? 1 : 0}`;
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    setFusionHash(Math.abs(hash).toString(16).toUpperCase());
  }, [headId, bodyId, scale, offsetX, offsetY, usePaletteSwap]);

  // 3. Core Fusion Logic
  const drawFusion = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !headImage || !bodyImage) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Canvas dimensions (simulating a 96x96 sprite scaled up)
    const W = canvas.width;
    const H = canvas.height;

    // --- STEP A: DRAW BODY (BASE) ---
    // Draw body to an offscreen canvas first to manipulate pixels if needed
    const bodyCanvas = document.createElement('canvas');
    bodyCanvas.width = W;
    bodyCanvas.height = H;
    const bCtx = bodyCanvas.getContext('2d');
    
    // Draw centered
    const baseScale = 2.5; // Base zoom for visibility
    bCtx.imageSmoothingEnabled = false;
    bCtx.drawImage(bodyImage, W/2 - (bodyImage.width * baseScale)/2, H/2 - (bodyImage.height * baseScale)/2, bodyImage.width * baseScale, bodyImage.height * baseScale);

    // --- PALETTE HARMONIZATION (THE ALGORITHM) ---
    if (usePaletteSwap) {
      const headCanvas = document.createElement('canvas');
      headCanvas.width = headImage.width;
      headCanvas.height = headImage.height;
      const hCtx = headCanvas.getContext('2d');
      hCtx.drawImage(headImage, 0, 0);
      
      const bodyData = bCtx.getImageData(0, 0, W, H);
      const headData = hCtx.getImageData(0, 0, headImage.width, headImage.height);
      
      // 1. Extract Head Colors (Simple clustering by Luminance)
      const headColors = [];
      for (let i = 0; i < headData.data.length; i += 4) {
        if (headData.data[i + 3] > 128) { // Only opaque pixels
          headColors.push([headData.data[i], headData.data[i+1], headData.data[i+2]]);
        }
      }
      
      // 2. Extract Body Colors
      // We map pixels based on relative luminance (light body pixel -> light head color)
      // This preserves shading while swapping hue.
      if (headColors.length > 0) {
        // Sort head colors by luminance
        headColors.sort((a, b) => {
            const lumA = 0.299*a[0] + 0.587*a[1] + 0.114*a[2];
            const lumB = 0.299*b[0] + 0.587*b[1] + 0.114*b[2];
            return lumA - lumB;
        });

        const pixels = bodyData.data;
        for (let i = 0; i < pixels.length; i += 4) {
          if (pixels[i+3] < 10) continue; // Skip transparent

          const r = pixels[i];
          const g = pixels[i+1];
          const b = pixels[i+2];
          
          // Calculate Body Pixel Luminance (0-1)
          const lum = (0.299*r + 0.587*g + 0.114*b) / 255;
          
          // Map to Head Palette Index
          // We use a non-linear mapping to preserve contrast
          const index = Math.floor(Math.pow(lum, contrastBoost) * (headColors.length - 1));
          const safeIndex = Math.min(Math.max(index, 0), headColors.length - 1);
          
          const newColor = headColors[safeIndex];
          
          // Blend slightly with original to avoid harsh banding (0.8 strength)
          pixels[i] = newColor[0] * 0.9 + r * 0.1;
          pixels[i+1] = newColor[1] * 0.9 + g * 0.1;
          pixels[i+2] = newColor[2] * 0.9 + b * 0.1;
        }
        bCtx.putImageData(bodyData, 0, 0);
      }
    }

    // Draw processed body to main canvas
    ctx.drawImage(bodyCanvas, 0, 0);

    // --- STEP B: DRAW HEAD (PARTS COMPOSITION) ---
    ctx.save();
    
    // Apply transformations (Anchor Points logic)
    const headW = headImage.width * baseScale * scale;
    const headH = headImage.height * baseScale * scale;
    const centerX = W/2 + offsetX;
    const centerY = H/2 + offsetY;

    ctx.translate(centerX, centerY);
    if (flipHead) ctx.scale(-1, 1);
    
    // Edge-Aware Seam Smoothing (Feathering)
    if (feathering > 0) {
       ctx.shadowBlur = feathering;
       // Average color of head for shadow
       ctx.shadowColor = "rgba(0,0,0,0.3)"; 
    }

    ctx.drawImage(headImage, -headW/2, -headH/2, headW, headH);
    ctx.restore();

  }, [headImage, bodyImage, scale, offsetX, offsetY, flipHead, usePaletteSwap, feathering, contrastBoost]);

  useEffect(() => {
    requestAnimationFrame(drawFusion);
  }, [drawFusion]);


  // Handlers
  const handleRandomize = () => {
    setHeadId(Math.floor(Math.random() * 151) + 1);
    setBodyId(Math.floor(Math.random() * 151) + 1);
  };

  const handleSwap = () => {
      const temp = headId;
      setHeadId(bodyId);
      setBodyId(temp);
  }

  const handleCanvasMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    setDragStart({ x: e.clientX - offsetX, y: e.clientY - offsetY });
    setIsDragging(true);
  };

  const handleCanvasMouseMove = (e) => {
    if (!isDragging) return;
    setOffsetX(e.clientX - dragStart.x);
    setOffsetY(e.clientY - dragStart.y);
  };

  const handleCanvasMouseUp = () => {
    setIsDragging(false);
  };

  const downloadImage = () => {
      const link = document.createElement('a');
      link.download = `fusion_${fusionHash}.png`;
      link.href = canvasRef.current.toDataURL();
      link.click();
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 font-sans selection:bg-purple-500 selection:text-white">
      <div className="max-w-6xl mx-auto p-4 md:p-8">
        
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              DNA Splicer
            </h1>
            <p className="text-slate-400 text-sm mt-1">Procedural Pokémon Fusion Engine</p>
          </div>
          <div className="flex gap-2">
            <div className="px-3 py-1 bg-slate-800 rounded-full border border-slate-700 flex items-center gap-2 text-xs font-mono text-purple-300">
              <Hash size={12} />
              {fusionHash || 'INIT'}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* LEFT COLUMN: Controls */}
          <div className="lg:col-span-4 space-y-6">
            
            {/* Input Selection */}
            <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700 backdrop-blur-sm">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
                  <Layers size={16} /> genetic sources
                </h2>
                <button 
                  onClick={handleRandomize}
                  className="text-xs bg-purple-600 hover:bg-purple-500 text-white px-3 py-1 rounded-md transition-colors flex items-center gap-1"
                >
                  <Shuffle size={12} /> Randomize
                </button>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-slate-700 rounded-lg flex items-center justify-center overflow-hidden border border-slate-600">
                    {headImage && <img src={headImage.src} alt="Head" className="w-full h-full object-contain" />}
                  </div>
                  <div className="flex-1">
                    <label className="text-xs text-slate-500 block mb-1">Head (Top Layer)</label>
                    <input 
                      type="number" 
                      min="1" max="898" 
                      value={headId}
                      onChange={(e) => setHeadId(Number(e.target.value))}
                      className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm focus:border-purple-500 outline-none"
                    />
                  </div>
                </div>

                <div className="flex justify-center">
                    <button onClick={handleSwap} className="p-2 hover:bg-slate-700 rounded-full text-slate-500 transition-colors">
                        <ArrowRightLeft size={16} />
                    </button>
                </div>

                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-slate-700 rounded-lg flex items-center justify-center overflow-hidden border border-slate-600">
                    {bodyImage && <img src={bodyImage.src} alt="Body" className="w-full h-full object-contain" />}
                  </div>
                  <div className="flex-1">
                    <label className="text-xs text-slate-500 block mb-1">Body (Base Layer)</label>
                    <input 
                      type="number" 
                      min="1" max="898" 
                      value={bodyId}
                      onChange={(e) => setBodyId(Number(e.target.value))}
                      className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 text-sm focus:border-purple-500 outline-none"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Anchoring & Placement */}
            <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700">
              <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4 flex items-center gap-2">
                <Move size={16} /> Part Composition
              </h2>
              
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-400">Head Scale</span>
                    <span className="text-slate-500">{scale.toFixed(2)}x</span>
                  </div>
                  <input 
                    type="range" min="0.5" max="2.0" step="0.05"
                    value={scale}
                    onChange={(e) => setScale(Number(e.target.value))}
                    className="w-full accent-purple-500 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div className="flex items-center gap-2">
                    <input 
                        type="checkbox" 
                        id="flipHead"
                        checked={flipHead}
                        onChange={(e) => setFlipHead(e.target.checked)}
                        className="rounded bg-slate-700 border-slate-600 text-purple-600 focus:ring-purple-500"
                    />
                    <label htmlFor="flipHead" className="text-sm text-slate-400 select-none cursor-pointer">Mirror Head Sprite</label>
                </div>
                
                <p className="text-xs text-slate-500 italic mt-2">
                    Tip: Drag the head on the canvas to adjust anchor points.
                </p>
              </div>
            </div>

            {/* Coloring Algorithms */}
            <div className="bg-slate-800/50 p-5 rounded-xl border border-slate-700">
              <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider mb-4 flex items-center gap-2">
                <Palette size={16} /> Palette Harmonization
              </h2>

              <div className="space-y-4">
                 <div className="flex items-center justify-between p-3 bg-slate-900 rounded-lg border border-slate-700 cursor-pointer" onClick={() => setUsePaletteSwap(!usePaletteSwap)}>
                    <span className="text-sm text-slate-300">Apply Palette Swap</span>
                    <div className={`w-10 h-5 rounded-full relative transition-colors ${usePaletteSwap ? 'bg-purple-600' : 'bg-slate-600'}`}>
                        <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${usePaletteSwap ? 'left-6' : 'left-1'}`} />
                    </div>
                 </div>

                 {usePaletteSwap && (
                    <div className="space-y-4 pt-2 border-t border-slate-700/50">
                        <div>
                            <div className="flex justify-between text-xs mb-1">
                                <span className="text-slate-400">Contrast Preservation</span>
                                <span className="text-slate-500">{contrastBoost}</span>
                            </div>
                            <input 
                                type="range" min="0.5" max="3.0" step="0.1"
                                value={contrastBoost}
                                onChange={(e) => setContrastBoost(Number(e.target.value))}
                                className="w-full accent-purple-500 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                            />
                        </div>
                    </div>
                 )}

                 <div>
                    <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400 flex items-center gap-1"><Eraser size={10} /> Seam Feathering</span>
                        <span className="text-slate-500">{feathering}px</span>
                    </div>
                    <input 
                        type="range" min="0" max="10" step="0.5"
                        value={feathering}
                        onChange={(e) => setFeathering(Number(e.target.value))}
                        className="w-full accent-purple-500 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                    />
                </div>
              </div>
            </div>

          </div>

          {/* RIGHT COLUMN: Canvas */}
          <div className="lg:col-span-8 flex flex-col h-full">
            <div className="flex-1 bg-slate-800 rounded-xl border border-slate-700 shadow-2xl relative overflow-hidden flex flex-col items-center justify-center p-8">
              
              {/* Background Grid Pattern */}
              <div className="absolute inset-0 opacity-10" 
                   style={{backgroundImage: 'radial-gradient(#64748b 1px, transparent 1px)', backgroundSize: '20px 20px'}}>
              </div>

              {loading && (
                <div className="absolute inset-0 bg-slate-900/80 z-20 flex items-center justify-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
                </div>
              )}

              {/* The Workbench */}
              <div className="relative group">
                <canvas
                    ref={canvasRef}
                    width={400}
                    height={400}
                    onMouseDown={handleCanvasMouseDown}
                    onMouseMove={handleCanvasMouseMove}
                    onMouseUp={handleCanvasMouseUp}
                    onMouseLeave={handleCanvasMouseUp}
                    className={`bg-slate-900 rounded-lg shadow-lg border-2 border-slate-700 cursor-move transition-shadow duration-300 ${isDragging ? 'shadow-purple-500/20 border-purple-500/50' : ''}`}
                    style={{ imageRendering: 'pixelated' }}
                />
                
                {/* Hints */}
                <div className="absolute -bottom-8 left-0 right-0 text-center opacity-0 group-hover:opacity-100 transition-opacity">
                    <span className="text-xs text-slate-500 bg-slate-900/90 px-2 py-1 rounded">Drag to adjust head position</span>
                </div>
              </div>

              {/* Action Bar */}
              <div className="mt-8 flex gap-4">
                <button 
                  onClick={downloadImage}
                  className="bg-purple-600 hover:bg-purple-500 text-white px-6 py-2 rounded-lg font-medium shadow-lg shadow-purple-900/20 transition-all flex items-center gap-2 active:scale-95"
                >
                  <Download size={18} /> Export Subject
                </button>
              </div>

            </div>
            
            {/* Debug/Info Panel */}
            <div className="mt-4 p-4 rounded-lg border border-slate-700/50 bg-slate-800/30 text-xs text-slate-500 font-mono">
                <p>STATUS: {loading ? 'SYNTHESIZING...' : 'STABLE'}</p>
                <p>ANCHOR_OFFSET: X:{offsetX} Y:{offsetY}</p>
                <p>PALETTE_MODE: {usePaletteSwap ? 'LUMINANCE_MAPPING' : 'ORIGINAL'}</p>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}