import io, os, json, hashlib, base64, math, subprocess, sys
from typing import Tuple, List, Literal
from datetime import datetime
import pathlib
import colorsys

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from PIL import Image, ImageStat, ImageOps, ImageFilter, ImageDraw, ImageFont

# -------------------- FastAPI app --------------------
app = FastAPI(title="Fusion Art Simple (Improved)", version="0.2.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
CREATURES = [
    {"id": "flamara",   "name": "Flamara"},
    {"id": "aquaphin",  "name": "Aquaphin"},
    {"id": "verdantle", "name": "Verdantle"},
    {"id": "sparyx",    "name": "Sparyx"},
]

# Caching dir
CACHE_DIR = pathlib.Path(os.path.dirname(__file__)) / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Additional module roots
VGG_OUTPUT = pathlib.Path(os.path.dirname(__file__)) / "VGG" / "output_images"
GAN_ROOT    = pathlib.Path(os.path.dirname(__file__)) / "GAN"
GALLERY_DIR = pathlib.Path(os.path.dirname(__file__)).parent / "gallery"
DIFFUSION_PIPE = None

# -------------------- Models --------------------
FusionMethod = Literal[
    "half", "headbody", "leftright", "maskblend",
    "offset", "diag", "graphcut", "pyramid", "parts3"
]

StyleName = Literal["illustrative", "realistic-soft", "sketch"]

class FuseSpec(BaseModel):
    parents: Tuple[str, str]
    seed: int = 0
    method: FusionMethod = "half"
    harmonize: bool = True
    harm_amount: float = Field(0.35, ge=0.0, le=1.0)
    feather_px: int = Field(6, ge=0)

class StyleSpec(BaseModel):
    parents: Tuple[str, str]
    seed: int = 0
    style: StyleName = "illustrative"
    method: FusionMethod = "half"
    harmonize: bool = True
    harm_amount: float = Field(0.35, ge=0.0, le=1.0)
    feather_px: int = Field(6, ge=0)

# -------------------- IO helpers --------------------

def _ensure_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA") if img.mode != "RGBA" else img

def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def png_b64(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(to_png_bytes(img)).decode("ascii")

def pil_to_dataurl_gif(imgs, duration_ms=120, loop=0) -> str:
    frames = [im.convert("RGBA") for im in imgs]
    # use a global palette for GIF
    pal_frames = [fr.convert("P", palette=Image.ADAPTIVE, colors=256, dither=Image.Dither.NONE) for fr in frames]
    bio = io.BytesIO()
    pal_frames[0].save(
        bio, format="GIF",
        save_all=True,
        append_images=pal_frames[1:],
        duration=duration_ms,
        loop=loop,
        transparency=255,
        disposal=2
    )
    return "data:image/gif;base64," + base64.b64encode(bio.getvalue()).decode("ascii")

def decode_data_uri(data_url: str) -> Image.Image:
    if not isinstance(data_url, str) or not data_url.startswith("data:image"):
        raise TypeError("expected data URL string")
    _, b64 = data_url.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

def any_to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGBA")
    if isinstance(x, str) and x.startswith("data:image"):
        return decode_data_uri(x)
    raise TypeError("unsupported image type")

def cache_paths(h: str):
    base = CACHE_DIR / h
    return {
        "spec": base.with_suffix(".json"),
        "base": base.with_suffix(".base.png"),
        "styled": base.with_suffix(".styled.png"),
    }

def save_cache(h: str, spec: dict, base_img: Image.Image = None, styled_img: Image.Image = None):
    p = cache_paths(h)
    payload = {"hash": h, "spec": spec, "ts": datetime.utcnow().isoformat() + "Z"}
    p["spec"].write_text(json.dumps(payload, indent=2))
    if base_img is not None:
        base_img.save(p["base"], format="PNG")
    if styled_img is not None:
        styled_img.save(p["styled"], format="PNG")

def b64_of_file(p: pathlib.Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")

def load_cache(h: str) -> dict | None:
    """Load a cached record and return dict with data-URLs for base/styled if present."""
    paths = cache_paths(h)
    if not paths["spec"].exists():
        return None
    try:
        meta = json.loads(paths["spec"].read_text())
    except Exception:
        meta = {"hash": h, "spec": {}}
    rec = {"hash": h, "spec": meta.get("spec", {}), "ts": meta.get("ts")}
    if paths["base"].exists():
        rec["base"] = b64_of_file(paths["base"])
    if paths["styled"].exists():
        rec["styled"] = b64_of_file(paths["styled"])
    return rec

def list_recent_images(root: pathlib.Path, limit: int = 12):
    """Return list of images (data URLs) sorted by mtime descending."""
    if not root.exists():
        return []
    exts = (".png", ".jpg", ".jpeg")
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:max(1, min(limit, 40))]
    out = []
    for p in files:
        try:
            out.append({"name": str(p.relative_to(root)), "image": b64_of_file(p)})
        except Exception:
            continue
    return out

def _load_diffusion_pipe():
    """Lazy-load Stable Diffusion Img2Img pipeline."""
    global DIFFUSION_PIPE
    if DIFFUSION_PIPE is not None:
        return DIFFUSION_PIPE
    try:
        from diffusers import StableDiffusionImg2ImgPipeline  # type: ignore
        import torch  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"diffusion dependencies missing: {e}")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")  # type: ignore[attr-defined]
    DIFFUSION_PIPE = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    return DIFFUSION_PIPE

def run_subprocess(cmd: list[str], cwd: pathlib.Path) -> tuple[int, str, str]:
    """Run a child process and capture output."""
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

# -------------------- Upscalers & backgrounds --------------------

def upscale_pxnn(img: Image.Image, scale: int) -> Image.Image:
    """Crisp integer nearest-neighbor (perfect for pixel art)."""
    img = _ensure_rgba(img)
    scale = max(1, int(scale))
    return img.resize((img.width*scale, img.height*scale), resample=Image.NEAREST)

def upscale_scale2x(img: Image.Image, scale: int) -> Image.Image:
    """
    Scale2x-like: a single 2x pass; if scale>2, finish with NN.
    """
    img = _ensure_rgba(img)
    def scale2x_pass(arr):
        h, w, c = arr.shape
        out = np.zeros((h*2, w*2, c), dtype=np.uint8)
        P = arr
        rgb = P[..., :3].astype(np.int16)
        a = np.pad(rgb, ((1,1),(1,1),(0,0)), mode='edge')
        A = a[:-2,1:-1]; B = a[:-2,2:]; D = a[1:-1,:-2]; E = a[1:-1,1:-1]
        F = a[1:-1,2:]; H = a[2:,:-2]
        cond = (np.any(D!=F,axis=2)) & (np.any(B!=H,axis=2))
        TL = np.where(cond[...,None] & np.any(D==B,axis=2,keepdims=True), D, E)
        TR = np.where(cond[...,None] & np.any(B==F,axis=2,keepdims=True), F, E)
        BL = np.where(cond[...,None] & np.any(D==H,axis=2,keepdims=True), D, E)
        BR = np.where(cond[...,None] & np.any(H==F,axis=2,keepdims=True), F, E)
        Aout = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        Aout[0::2,0::2] = TL.astype(np.uint8)
        Aout[0::2,1::2] = TR.astype(np.uint8)
        Aout[1::2,0::2] = BL.astype(np.uint8)
        Aout[1::2,1::2] = BR.astype(np.uint8)
        alpha = np.repeat(np.repeat(P[...,3:4], 2, axis=0), 2, axis=1)
        return np.concatenate([Aout, alpha], axis=2)

    arr = np.array(img)
    pass2x = scale2x_pass(arr)
    out = Image.fromarray(pass2x, mode="RGBA")
    if scale > 2:
        out = out.resize((img.width*scale, img.height*scale), Image.NEAREST)
    return out

def upscale_lanczos_edge(img: Image.Image, scale: int) -> Image.Image:
    """
    Smooth upscale without smearing edges:
    - upscale Lanczos
    - detect edges on luminance
    - unsharp on edges only
    """
    img = _ensure_rgba(img)
    scale = max(1, int(scale))
    big = img.resize((img.width*scale, img.height*scale), Image.LANCZOS)

    lum = big.convert("L")
    edges = lum.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=max(0.5, scale/3)))
    sharp = big.filter(ImageFilter.UnsharpMask(radius=max(1.0, scale/2), percent=150, threshold=0))
    mask = edges.point(lambda p: min(255, int(p*1.5)))
    return Image.composite(sharp, big, mask)

def render_background(canvas_size, kind="none"):
    w, h = canvas_size
    if kind == "none":
        return Image.new("RGBA", (w, h), (0,0,0,0))
    if kind == "sunset":
        top = np.array([255, 206, 145], dtype=np.float32)
        bot = np.array([120,  70, 160], dtype=np.float32)
        t = np.linspace(0,1,h, dtype=np.float32)[:,None]
        grad = (top*(1-t) + bot*t).astype(np.uint8)
        grad = np.repeat(grad, w, axis=1).reshape(h,w,3)
        im = Image.fromarray(grad, "RGB").convert("RGBA")
        return im
    if kind == "halo":
        bg = Image.new("RGBA", (w,h), (24,24,32,255))
        cx, cy = w/2, h/2
        Y, X = np.ogrid[:h,:w]
        d = np.sqrt((X-cx)**2 + (Y-cy)**2) / max(w,h)
        a = (255*np.clip(1 - d*1.6, 0, 1)).astype(np.uint8)
        glow = Image.new("RGBA",(w,h),(130,180,255,0))
        glow.putalpha(Image.fromarray(a, "L").filter(ImageFilter.GaussianBlur(radius=12)))
        bg.alpha_composite(glow)
        return bg
    return Image.new("RGBA", (w, h), (0,0,0,0))

def compose_on_bg(fg: Image.Image, bg_kind="none", pad=24, scale=4):
    """Center the sprite on a background with padding (pixel-art friendly upscale)."""
    fg = _ensure_rgba(fg)
    w = (fg.width + pad*2) * scale
    h = (fg.height + pad*2) * scale
    bg = render_background((w,h), bg_kind)
    up = upscale_pxnn(fg, scale)
    x = (w - up.width)//2
    y = (h - up.height)//2
    bg.alpha_composite(up, (x,y))
    return bg

def draw_card(fusion_img: Image.Image, title: str, subtitle: str, stats: dict | None = None):
    W, H = 800, 600
    card = Image.new("RGBA", (W, H), (18,18,20,255))
    head = Image.new("RGBA", (W, 90), (28,28,36,255))
    card.alpha_composite(head, (0,0))
    draw = ImageDraw.Draw(card)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 24)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font_title = font_sub = font_small = ImageFont.load_default()

    draw.text((24, 22), title, fill=(240,240,255,255), font=font_title)
    draw.text((24, 60), subtitle, fill=(190,190,205,255), font=font_sub)

    img_area = Image.new("RGBA", (W-80, H-220), (0,0,0,0))
    sprite = compose_on_bg(fusion_img, bg_kind="halo", pad=12, scale=4)
    sx = (img_area.width - sprite.width)//2
    sy = (img_area.height - sprite.height)//2
    img_area.alpha_composite(sprite, (sx, max(0,sy)))
    frame = Image.new("RGBA", (img_area.width+8, img_area.height+8), (255,255,255,20))
    card.alpha_composite(frame, (36-4, 120-4))
    card.alpha_composite(img_area, (36, 120))

    foot_y = H - 80
    draw.rectangle((0, foot_y, W, H), fill=(28,28,36,255))
    if stats:
        x = 24
        for k in ["HP","ATK","DEF","SPD"]:
            val = stats.get(k, "—")
            draw.text((x, foot_y+22), f"{k}: {val}", fill=(220,220,235,255), font=font_small)
            x += 140
    return card

def animate_idle(img: Image.Image, frames: int = 8):
    base = _ensure_rgba(img)
    w, h = base.size
    out = []
    for t in range(frames):
        phase = 2*np.pi * (t/frames)
        dy = int(round(np.sin(phase)*2))
        scale = 1.0 + 0.02*np.sin(phase+np.pi/2)
        frame = Image.new("RGBA",(w,h),(0,0,0,0))
        # shadow
        shadow = Image.new("RGBA",(w,h),(0,0,0,0))
        s_el = Image.new("RGBA",(w, int(h*0.1)), (0,0,0,120))
        s_el = s_el.filter(ImageFilter.GaussianBlur(radius=4))
        shadow.alpha_composite(s_el, (0, int(h*0.85)))
        frame.alpha_composite(shadow)
        # sprite
        sw, sh = w, int(h*scale)
        sp = base.resize((sw, sh), Image.NEAREST)
        y = max(0, (h - sh)//2 + dy)
        frame.alpha_composite(sp, (0, y))
        out.append(frame)
    return out

# -------------------- Color utilities --------------------
def avg_color(img: Image.Image) -> Tuple[int, int, int]:
    r, g, b, a = img.split()
    if a.getbbox() is None:
        stat = ImageStat.Stat(Image.merge("RGB", (r, g, b)))
    else:
        stat = ImageStat.Stat([r, g, b], mask=a)
    means = stat.mean
    return tuple(int(v) for v in means[:3])

def extract_palette(img: Image.Image, k: int = 5) -> List[Tuple[int, int, int]]:
    a = img.split()[-1]
    box = a.getbbox()
    src = img if box is None else img.crop(box)

    comp = Image.alpha_composite(Image.new("RGBA", src.size, (0, 0, 0, 255)), src).convert("RGB")
    q = comp.quantize(colors=max(2, min(16, k*3)), method=Image.FASTOCTREE)
    colors_info = q.getcolors(q.width * q.height) or []
    pal_list = q.getpalette() or []

    out: List[Tuple[int, int, int]] = []
    seen = set()
    for _, idx in sorted(colors_info, reverse=True):
        if isinstance(idx, int):
            r = pal_list[3*idx]     if 3*idx     < len(pal_list) else 0
            g = pal_list[3*idx + 1] if 3*idx + 1 < len(pal_list) else 0
            b = pal_list[3*idx + 2] if 3*idx + 2 < len(pal_list) else 0
            rgb = (int(r), int(g), int(b))
        elif isinstance(idx, (tuple, list)) and len(idx) >= 3:
            rgb = (int(idx[0]), int(idx[1]), int(idx[2]))
        else:
            continue
        if rgb not in seen:
            seen.add(rgb)
            out.append(rgb)
        if len(out) >= k:
            break

    if not out:
        out = [avg_color(img)]
    return out

def rgb_to_hsv(r, g, b):
    return colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)

def hsv_to_rgb(h, s, v):
    R, G, B = colorsys.hsv_to_rgb(h, s, v)
    return int(R*255), int(G*255), int(B*255)

def harmonize_toward(img: Image.Image, target_rgb: Tuple[int,int,int], amount: float) -> Image.Image:
    if amount <= 0: return img
    h_t, s_t, v_t = rgb_to_hsv(*target_rgb)
    out = Image.new("RGBA", img.size, (0,0,0,0))
    src = img.load(); dst = out.load()
    for y in range(img.height):
        for x in range(img.width):
            r,g,b,a = src[x,y]
            if a == 0: dst[x,y]=(0,0,0,0); continue
            h,s,v = rgb_to_hsv(r,g,b)
            h2 = (1-amount)*h + amount*h_t
            s2 = (1-amount)*s + amount*s_t
            v2 = (1-amount*0.15)*v + (amount*0.15)*v_t
            R,G,B = hsv_to_rgb(h2,s2,v2)
            dst[x,y]=(R,G,B,a)
    return out

# -------------------- Fusion utilities --------------------
def _band_cost(imgA: Image.Image, imgB: Image.Image, band: tuple) -> np.ndarray:
    x0, x1 = band
    A = np.array(imgA.convert("RGB"))[:, x0:x1, :].astype(np.float32)
    B = np.array(imgB.convert("RGB"))[:, x0:x1, :].astype(np.float32)
    return np.sqrt(((A - B) ** 2).sum(axis=2))  # H x Wband

def _min_cost_seam(cost: np.ndarray) -> np.ndarray:
    H, W = cost.shape
    dp = cost.copy()
    back = np.zeros((H, W), dtype=np.int16)
    for y in range(1, H):
        for x in range(W):
            prevs = []
            idxs = []
            for dx in (-1, 0, 1):
                xx = x + dx
                if 0 <= xx < W:
                    prevs.append(dp[y-1, xx])
                    idxs.append(xx)
            j = int(np.argmin(prevs))
            dp[y, x] += prevs[j]
            back[y, x] = idxs[j]
    seam = np.zeros(H, dtype=np.int16)
    seam[-1] = int(np.argmin(dp[-1]))
    for y in range(H-2, -1, -1):
        seam[y] = back[y+1, seam[y+1]]
    return seam

def _mask_from_seam(w: int, h: int, x0: int, seam_cols: np.ndarray, feather: int) -> Image.Image:
    M = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        seam_x = x0 + int(seam_cols[y])
        M[y, :seam_x] = 255.0
    mask = Image.fromarray(M.astype(np.uint8), mode="L")
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(max(0.01, feather/2)))
    return mask

def fuse_offset(a: Image.Image, b: Image.Image, feather_px: int, vertical=True, frac=0.5) -> Image.Image:
    w, h = a.size
    out = Image.new("RGBA", (w, h), (0,0,0,0))
    if vertical:
        cut = int(w * frac)
        out.paste(a.crop((0,0,cut,h)), (0,0), a.crop((0,0,cut,h)))
        band = b.crop((cut,0,w,h)).copy()
        if feather_px>0:
            m = Image.new("L", (band.width, band.height), 255).filter(ImageFilter.GaussianBlur(max(0.01, feather_px/2)))
            band.putalpha(m)
        out.alpha_composite(band, (cut,0))
    else:
        cut = int(h * frac)
        out.paste(a.crop((0,0,w,cut)), (0,0), a.crop((0,0,w,cut)))
        band = b.crop((0,cut,w,h)).copy()
        if feather_px>0:
            m = Image.new("L", (band.width, band.height), 255).filter(ImageFilter.GaussianBlur(max(0.01, feather_px/2)))
            band.putalpha(m)
        out.alpha_composite(band, (0,cut))
    return out

def fuse_diag(a: Image.Image, b: Image.Image, feather_px: int, reverse=False) -> Image.Image:
    w, h = a.size
    M = np.fromfunction(lambda yy, xx: (xx + (h-yy)) if reverse else (xx + yy), (h, w), dtype=int)
    M = (M - M.min()) / (M.max() - M.min() + 1e-6)
    mask = Image.fromarray((M*255).astype(np.uint8), mode="L")
    if feather_px>0:
        mask = mask.filter(ImageFilter.GaussianBlur(max(0.01, feather_px/2)))
    return Image.composite(a, b, mask)

def fuse_graphcut(a: Image.Image, b: Image.Image, feather_px: int, band_half=10) -> Image.Image:
    w, h = a.size
    cx = w // 2
    x0 = max(0, cx - band_half)
    x1 = min(w, cx + band_half + 1)
    cost = _band_cost(a, b, (x0, x1))
    seam_cols = _min_cost_seam(cost)
    mask = _mask_from_seam(w, h, x0, seam_cols, feather_px)
    return Image.composite(a, b, mask)

def radial_mask(size: Tuple[int,int], center=(0.5,0.45), sigma=0.35) -> Image.Image:
    w,h = size
    cx,cy = int(center[0]*w), int(center[1]*h)
    mask = Image.new("L", (w,h), 0)
    px = mask.load()
    for y in range(h):
        for x in range(w):
            dx = (x-cx)/max(1,w); dy = (y-cy)/max(1,h)
            d = math.hypot(dx,dy)
            v = math.exp(-(d*d)/(2*sigma*sigma))
            px[x,y] = int(v*255)
    return mask.filter(ImageFilter.GaussianBlur(2))

def _alpha_from_mask(mask_L: Image.Image) -> Image.Image:
    if mask_L.mode != "L":
        mask_L = mask_L.convert("L")
    return mask_L

def _laplacian_pyramid_blend(A: Image.Image, B: Image.Image, mask_L: Image.Image, levels: int = 4) -> Image.Image:
    a = np.array(A.convert("RGB"), dtype=np.float32) / 255.0
    b = np.array(B.convert("RGB"), dtype=np.float32) / 255.0
    m = np.array(_alpha_from_mask(mask_L), dtype=np.float32) / 255.0
    m = np.repeat(m[..., None], 3, axis=2)

    def gaussian(img):
        return np.array(
            Image.fromarray(np.clip(img*255,0,255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)),
            dtype=np.float32
        )/255.0

    def build_pyramids(im, L):
        G = [im]
        for _ in range(L):
            G.append(gaussian(G[-1]))
        P = [G[i] - G[i+1] for i in range(L)]
        P.append(G[-1])
        return P

    La = build_pyramids(a, levels)
    Lb = build_pyramids(b, levels)
    Lm = build_pyramids(m, levels)

    Lblend = [(Lm[i] * La[i] + (1 - Lm[i]) * Lb[i]) for i in range(levels+1)]
    out = Lblend[-1]
    for i in range(levels-1, -1, -1):
        out = out + Lblend[i]

    out = np.clip(out, 0, 1)
    rgb = (out * 255).astype(np.uint8)
    alpha = np.maximum(np.array(A.split()[-1]), np.array(B.split()[-1]))
    return Image.merge("RGBA", (Image.fromarray(rgb[...,0]),
                                Image.fromarray(rgb[...,1]),
                                Image.fromarray(rgb[...,2]),
                                Image.fromarray(alpha)))

def fuse_pyramid(a: Image.Image, b: Image.Image, feather_px: int) -> Image.Image:
    mask = radial_mask(a.size, center=(0.52,0.48), sigma=0.40)
    return _laplacian_pyramid_blend(a, b, mask_L=mask, levels=4)

def fuse_parts3(a: Image.Image, b: Image.Image, feather_px: int) -> Image.Image:
    w, h = a.size
    head_h   = min(22, h//3)
    torso_h  = min(26, h//3 + 6)
    legs_y   = head_h + torso_h
    out = Image.new("RGBA", (w, h), (0,0,0,0))

    head = a.crop((0,0,w,head_h))
    torso= b.crop((0,head_h,w,head_h+torso_h))
    legs = a.crop((0,legs_y,w,h))

    out.paste(torso,(0,head_h),torso)
    out.paste(head,(0,0),head)
    out.paste(legs,(0,legs_y),legs)

    if feather_px>0:
        for y0 in (head_h-1, legs_y-1):
            y0 = max(0, y0); y1 = min(h, y0 + feather_px*2+1)
            band = out.crop((0,y0,w,y1)).copy()
            m = Image.new("L", band.size, 255).filter(ImageFilter.GaussianBlur(max(0.01, feather_px/2)))
            band.putalpha(m)
            out.alpha_composite(band, (0,y0))
    return out

def fuse_half(a: Image.Image, b: Image.Image, feather_px: int) -> Image.Image:
    w, h = a.size
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    upper = a.crop((0, 0, w, h // 2))
    lower = b.crop((0, h // 2, w, h))
    out.paste(upper, (0, 0), upper)
    out.paste(lower, (0, h // 2), lower)
    if feather_px > 0:
        y0 = max(0, (h // 2) - feather_px)
        y1 = min(h, (h // 2) + feather_px)
        band = b.crop((0, y0, w, y1)).copy()
        mask = Image.new("L", band.size, 255).filter(ImageFilter.GaussianBlur(radius=max(0.01, feather_px / 2)))
        band.putalpha(mask)
        out.alpha_composite(band, (0, y0))
    return out

def fuse_leftright(a: Image.Image, b: Image.Image, feather_px: int) -> Image.Image:
    w, h = a.size
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    left = a.crop((0, 0, w // 2, h))
    right = b.crop((w // 2, 0, w, h))
    out.paste(left, (0, 0), left)
    out.paste(right, (w // 2, 0), right)
    if feather_px > 0:
        x0 = max(0, (w // 2) - feather_px)
        x1 = min(w, (w // 2) + feather_px)
        band = b.crop((x0, 0, x1, h)).copy()
        mask = Image.new("L", band.size, 255).filter(ImageFilter.GaussianBlur(radius=max(0.01, feather_px / 2)))
        band.putalpha(mask)
        out.alpha_composite(band, (x0, 0))
    return out

def fuse_headbody(a: Image.Image, b: Image.Image, feather_px: int) -> Image.Image:
    w, h = a.size
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    head_h = min(28, h // 2)
    head = a.crop((0, 0, w, head_h))
    body_top = max(0, head_h - 8)
    body = b.crop((0, body_top, w, h))
    out.paste(body, (0, body_top), body)
    out.paste(head, (0, 0), head)
    if feather_px > 0:
        y0 = body_top
        y1 = min(h, body_top + feather_px * 2)
        band = body.crop((0, y0, w, y1)).copy()
        mask = Image.new("L", band.size, 255).filter(ImageFilter.GaussianBlur(radius=max(0.01, feather_px / 2)))
        band.putalpha(mask)
        out.alpha_composite(band, (0, y0))
    return out

def fuse_maskblend(a: Image.Image, b: Image.Image) -> Image.Image:
    return Image.composite(a, b, radial_mask(a.size, center=(0.5,0.42), sigma=0.38))

def build_fusion(a: Image.Image, b: Image.Image, method: FusionMethod, feather_px: int) -> Image.Image:
    if method == "half":       return fuse_half(a,b,feather_px)
    if method == "headbody":   return fuse_headbody(a,b,feather_px)
    if method == "leftright":  return fuse_leftright(a,b,feather_px)
    if method == "maskblend":  return fuse_maskblend(a,b)
    if method == "offset":     return fuse_offset(a,b,feather_px, vertical=True, frac=0.45)
    if method == "diag":       return fuse_diag(a,b,feather_px, reverse=False)
    if method == "graphcut":   return fuse_graphcut(a,b,feather_px, band_half=12)
    if method == "pyramid":    return fuse_pyramid(a,b,feather_px)
    if method == "parts3":     return fuse_parts3(a,b,feather_px)
    return fuse_half(a,b,feather_px)

def unified_palette_color(a: Image.Image, b: Image.Image) -> Tuple[int, int, int]:
    def _rgb3(c):
        if isinstance(c, (tuple, list)) and len(c) >= 3:
            return (int(c[0]), int(c[1]), int(c[2]))
        if isinstance(c, int):
            return (c, c, c)
        return (0, 0, 0)
    ca = _rgb3(avg_color(a))
    cb = _rgb3(avg_color(b))
    pa = [ _rgb3(c) for c in extract_palette(a, k=4) ]
    pb = [ _rgb3(c) for c in extract_palette(b, k=4) ]
    pool = [ca, cb] + pa[:2] + pb[:2]
    if not pool:
        return (128,128,128)
    R = sum(c[0] for c in pool) / len(pool)
    G = sum(c[1] for c in pool) / len(pool)
    B = sum(c[2] for c in pool) / len(pool)
    return (int(R), int(G), int(B))

# -------------------- Styling --------------------
def stylize_filter(img: Image.Image, mode: StyleName) -> Image.Image:
    rgb = img.convert("RGB")
    if mode == "illustrative":
        out = ImageOps.posterize(rgb, 3).filter(ImageFilter.EDGE_ENHANCE_MORE)
    elif mode == "realistic-soft":
        out = rgb.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.DETAIL)
    else:  # sketch
        edges = rgb.convert("L").filter(ImageFilter.FIND_EDGES).point(lambda p: 255 - p)
        edges = ImageOps.autocontrast(edges)
        out = Image.merge("RGB", (edges, edges, edges))
    out = out.convert("RGBA")
    out.putalpha(img.split()[-1])
    return out

# -------------------- Core (shared) --------------------
def load_sprite(cid: str) -> Image.Image:
    p = os.path.join(ASSETS, f"{cid}.png")
    if not os.path.exists(p):
        raise FileNotFoundError(cid)
    return Image.open(p).convert("RGBA")

def _fuse_core(spec: FuseSpec) -> Image.Image:
    a = load_sprite(spec.parents[0])
    b = load_sprite(spec.parents[1])
    fused = build_fusion(a, b, spec.method, spec.feather_px)
    if spec.harmonize:
        target = unified_palette_color(a, b)
        fused = harmonize_toward(fused, target, spec.harm_amount)
    return fused

def _fuse_from_images(a_img: Image.Image, b_img: Image.Image, method: FusionMethod, harmonize: bool, harm_amount: float, feather_px: int) -> Image.Image:
    a = _ensure_rgba(a_img)
    b = _ensure_rgba(b_img)
    if a.size != b.size:
        target = (min(a.width, b.width), min(a.height, b.height))
        target = (max(8, target[0]), max(8, target[1]))
        a = a.resize(target, Image.NEAREST)
        b = b.resize(target, Image.NEAREST)
    fused = build_fusion(a, b, method, feather_px)
    if harmonize:
        target = unified_palette_color(a, b)
        fused = harmonize_toward(fused, target, harm_amount)
    return fused

# -------------------- Routes --------------------
@app.get("/health")
def health():
    return {"ok": True, "creatures": len(CREATURES)}

@app.get("/creatures")
def creatures():
    return CREATURES

@app.post("/fuse")
def fuse(spec: FuseSpec):
    try:
        fused = _fuse_core(spec)
    except FileNotFoundError as e:
        raise HTTPException(400, f"Unknown parent id: {e}") from e
    except Exception as e:
        raise HTTPException(400, f"fuse failed: {e}") from e

    h = hashlib.sha256(json.dumps(spec.model_dump(), sort_keys=True).encode()).hexdigest()[:16]
    try:
        save_cache(h, {"route": "fuse", **spec.model_dump()}, base_img=fused)
    except Exception:
        pass
    return {"hash": h, "image": png_b64(fused)}

@app.post("/style")
def style(spec: StyleSpec):
    try:
        fused = _fuse_core(FuseSpec(
            parents=spec.parents,
            seed=spec.seed,
            method=spec.method,
            harmonize=spec.harmonize,
            harm_amount=spec.harm_amount,
            feather_px=spec.feather_px
        ))
        try:
            import torch, torchvision.transforms as T  # optional if available
            t = T.ToTensor()(fused.convert("RGB")).unsqueeze(0)
            t = torch.clamp(t * 1.08, 0, 1)
            styled_rgb = T.ToPILImage()(t[0])
            styled = styled_rgb.convert("RGBA")
            styled.putalpha(fused.split()[-1])
            styled = stylize_filter(styled, spec.style)
        except Exception:
            styled = stylize_filter(fused, spec.style)
    except FileNotFoundError as e:
        raise HTTPException(400, f"Unknown parent id: {e}") from e
    except Exception as e:
        raise HTTPException(400, f"style failed: {e}") from e

    h = hashlib.sha256(json.dumps(spec.model_dump(), sort_keys=True).encode()).hexdigest()[:16]
    try:
        save_cache(h, {"route": "style", **spec.model_dump()}, base_img=fused, styled_img=styled)
    except Exception:
        pass
    return {"hash": h, "base": png_b64(fused), "styled": png_b64(styled)}

@app.post("/style_upload")
async def style_upload(
    imageA: UploadFile = File(...),
    imageB: UploadFile = File(...),
    seed: int = Form(0),
    method: FusionMethod = Form("half"),
    style: StyleName = Form("illustrative"),
    harmonize: bool = Form(True),
    harm_amount: float = Form(0.35),
    feather_px: int = Form(6),
):
    try:
        a_img = Image.open(io.BytesIO(await imageA.read())).convert("RGBA")
        b_img = Image.open(io.BytesIO(await imageB.read())).convert("RGBA")
        fused = _fuse_from_images(
            a_img, b_img,
            method=method,
            harmonize=harmonize,
            harm_amount=harm_amount,
            feather_px=feather_px
        )
        styled = stylize_filter(fused, style)
        h = hashlib.sha256(f"{seed}-{imageA.filename}-{imageB.filename}-{method}-{style}".encode()).hexdigest()[:16]
        try:
            save_cache(h, {"route": "style_upload", "method": method, "style": style}, base_img=fused, styled_img=styled)
        except Exception:
            pass
        return {"hash": h, "base": png_b64(fused), "styled": png_b64(styled)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"style_upload failed: {e}")

@app.get("/gallery")
def gallery(n: int = 12):
    items = []
    specs = sorted(CACHE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for sp in specs[:max(1, min(50, n))]:
        h = sp.stem
        paths = cache_paths(h)
        try:
            meta = json.loads(sp.read_text())
        except Exception:
            meta = {"hash": h, "spec": {}}
        rec = {"hash": h, "spec": meta.get("spec", {}), "ts": meta.get("ts")}
        if paths["styled"].exists():
            rec["styled"] = b64_of_file(paths["styled"])
        if paths["base"].exists():
            rec["base"] = b64_of_file(paths["base"])
        items.append(rec)
    return {"items": items}

@app.get("/image/{h}/{kind}")
def image(h: str, kind: Literal["base","styled"]):
    p = cache_paths(h).get(kind)
    if not p or not p.exists():
        raise HTTPException(404, "not found")
    return {"hash": h, "kind": kind, "image": b64_of_file(p)}

# -------------------- Module galleries (VGG / GAN) --------------------
@app.get("/vgg/results")
def vgg_results(n: int = 12):
    """Return recent VGG NST outputs from output_images/."""
    items = list_recent_images(VGG_OUTPUT, limit=n)
    return {"items": items, "count": len(items)}

@app.get("/gan/results")
def gan_results(n: int = 12):
    """Return recent GAN outputs across GAN_* folders (result subdirs)."""
    roots = [p for p in GAN_ROOT.glob("GAN_*") if p.is_dir()]
    collected = []
    for r in roots:
        result_dir = r / "result"
        collected.extend(list_recent_images(result_dir, limit=n))
    # trim and keep stable order
    collected = collected[:max(1, min(n, len(collected)))]
    return {"items": collected, "count": len(collected)}

# Allow the frontend to list available GAN style folders
@app.get("/gan/styles")
def gan_styles():
    names = [p.name for p in GAN_ROOT.glob("GAN_*") if p.is_dir()]
    return {"styles": sorted(names)}

# Kick off GAN generation (uses existing trained weights)
@app.post("/gan/run")
def gan_run(style: str = "GAN_Ukiyoe"):
    folder = GAN_ROOT / style
    script = folder / "cyclegan_gen.py"
    if not script.exists():
        raise HTTPException(404, f"GAN style '{style}' not found")
    code, out, err = run_subprocess([sys.executable, script.name], cwd=script.parent)
    if code != 0:
        raise HTTPException(status_code=500, detail=f"GAN generation failed: {err or out}")
    return {"ok": True, "stdout": out, "stderr": err}

# Kick off VGG NST batch
@app.post("/vgg/run")
def vgg_run():
    script = VGG_OUTPUT.parent / "style_transfer.py"
    if not script.exists():
        raise HTTPException(404, "VGG style_transfer.py not found")
    code, out, err = run_subprocess([sys.executable, script.name], cwd=script.parent)
    if code != 0:
        raise HTTPException(status_code=500, detail=f"VGG NST failed: {err or out}")
    return {"ok": True, "stdout": out, "stderr": err}

# -------------------- Diffusion (img2img) --------------------
@app.get("/diffusion/samples")
def diffusion_samples():
    if not GALLERY_DIR.exists():
        return {"files": []}
    files = [p.name for p in GALLERY_DIR.iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg") and p.is_file()]
    return {"files": sorted(files)}

@app.get("/diffusion/samples/{filename}")
def diffusion_sample_file(filename: str):
    p = GALLERY_DIR / filename
    if not p.exists():
        raise HTTPException(404, "file not found")
    return Response(content=p.read_bytes(), media_type="image/png")

@app.post("/diffusion/fuse")
async def diffusion_fuse(
    use_samples: bool = Form(False),
    filenameA: str | None = Form(None),
    filenameB: str | None = Form(None),
    prompt: str = Form("fusion creature combining both Pokémon in anime style"),
    strength: float = Form(0.75),
    guidance: float = Form(7.5),
    imageA: UploadFile | None = File(None),
    imageB: UploadFile | None = File(None),
):
    try:
        pipe = _load_diffusion_pipe()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"diffusion load failed: {e}")

    try:
        if use_samples:
            if not filenameA or not filenameB:
                raise HTTPException(400, "Both filenames required")
            a_img = Image.open((GALLERY_DIR / filenameA)).convert("RGB")
            b_img = Image.open((GALLERY_DIR / filenameB)).convert("RGB")
        else:
            if not imageA or not imageB:
                raise HTTPException(400, "Both images required")
            a_img = Image.open(io.BytesIO(await imageA.read())).convert("RGB")
            b_img = Image.open(io.BytesIO(await imageB.read())).convert("RGB")

        b_img = b_img.resize(a_img.size)
        result = pipe(
            prompt=prompt,
            image=a_img,
            strength=float(strength),
            guidance_scale=float(guidance)
        ).images[0]

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"diffusion fuse failed: {e}")

# -------------------- Export --------------------
class ExportSpec(BaseModel):
    hash: str | None = None          # optional: export from cache by hash
    parents: list[str] | None = None # or build fresh like /style
    method: str = "half"
    style: str | None = None
    harmonize: bool = True
    harm_amount: float = 0.35
    feather_px: int = 8
    scale: int = 4
    upscaler: str = "pxnn"           # pxnn | scale2x | lanczos_edge
    background: str = "none"         # none | sunset | halo
    card: bool = False               # if true, return card image
    animate: bool = False            # if true, return GIF bytes
    seed: int | None = 0

@app.post("/export")
def export(spec: ExportSpec):
    """
    Build (or fetch) a fusion and return a high-res PNG (or GIF if animate).
    This version FIXES the earlier 'Image object is not subscriptable' error by
    always normalizing to PIL images before further processing.
    """
    # 1) Acquire fused image(s)
    try:
        if spec.hash:
            rec = load_cache(spec.hash)
            if not rec:
                raise HTTPException(status_code=404, detail="hash not found")
            base_img = any_to_pil(rec["base"]) if rec.get("base") else None
            styled_img = any_to_pil(rec["styled"]) if rec.get("styled") else None
            fused = styled_img or base_img
            parents = rec.get("spec", {}).get("parents", ["?", "?"])
            title = "Fusion"
        else:
            parents = spec.parents or ["flamara", "aquaphin"]
            fuse_img = _fuse_core(FuseSpec(
                parents=(parents[0], parents[1]),
                seed=spec.seed or 0,
                method=spec.method,  # type: ignore[arg-type]
                harmonize=spec.harmonize,
                harm_amount=spec.harm_amount,
                feather_px=spec.feather_px
            ))
            if spec.style in ("illustrative", "realistic-soft", "sketch"):
                fused = stylize_filter(fuse_img, spec.style)  # styled
            else:
                fused = fuse_img
            base_img = fuse_img
            styled_img = fused if fused is not base_img else None
            title = "Fusion"
        if fused is None:
            raise RuntimeError("export: no fused image available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"export failed (acquire): {e}")

    # 2) Upscale
    try:
        upscaler = (spec.upscaler or "pxnn").lower()
        sc = max(2, int(spec.scale))
        if upscaler == "scale2x":
            up = upscale_scale2x(fused, sc)
        elif upscaler == "lanczos_edge":
            up = upscale_lanczos_edge(fused, sc)
        else:
            up = upscale_pxnn(fused, sc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"export failed (upscale): {e}")

    # 3) Background
    try:
        composed = compose_on_bg(up, bg_kind=spec.background, pad=16, scale=1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"export failed (background): {e}")

    # 4) Card or animation
    try:
        if spec.card:
            stats = {"HP": 70, "ATK": 85, "DEF": 65, "SPD": 80}
            card = draw_card(up, title=title, subtitle=f"{parents[0]} × {parents[1]}", stats=stats)
            return {"png": png_b64(card)}
        if spec.animate:
            frames = animate_idle(up, frames=8)
            return {"gif": pil_to_dataurl_gif(frames, duration_ms=120, loop=0)}
        return {"png": png_b64(composed)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"export failed (finalize): {e}")
