
import io, os, json, hashlib, base64, math
from typing import Tuple, List, Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageStat, ImageOps, ImageFilter
from datetime import datetime
import pathlib
import colorsys

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

# -------------------- Models --------------------
FusionMethod = Literal["half", "headbody", "leftright", "maskblend"]
StyleName    = Literal["illustrative", "realistic-soft", "sketch"]

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
def load_sprite(cid: str) -> Image.Image:
    p = os.path.join(ASSETS, f"{cid}.png")
    if not os.path.exists(p):
        raise FileNotFoundError(cid)
    return Image.open(p).convert("RGBA")

def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def png_b64(img: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(to_png_bytes(img)).decode()

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
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

# -------------------- Color utilities --------------------
def avg_color(img: Image.Image) -> Tuple[int, int, int]:
    """Average RGB over non-transparent pixels. Robust to empty masks."""
    r, g, b, a = img.split()
    if a.getbbox() is None:
        stat = ImageStat.Stat(Image.merge("RGB", (r, g, b)))
    else:
        stat = ImageStat.Stat([r, g, b], mask=a)  # must be list
    means = stat.mean
    return tuple(int(v) for v in means[:3])

def extract_palette(img: Image.Image, k: int = 5) -> List[Tuple[int, int, int]]:
    """
    Representative colors from visible pixels using quantize; robust against odd returns.
    Always returns a list of RGB triples.
    """
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
        else:
            if isinstance(idx, (tuple, list)) and len(idx) >= 3:
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
        band = b.crop((0, y0, w, y1)).copy()               # size = (w, y1 - y0)
        mask = Image.new("L", band.size, 255).filter(
            ImageFilter.GaussianBlur(radius=max(0.01, feather_px / 2))
        )
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
        band = b.crop((x0, 0, x1, h)).copy()               # size = (x1 - x0, h)
        mask = Image.new("L", band.size, 255).filter(
            ImageFilter.GaussianBlur(radius=max(0.01, feather_px / 2))
        )
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
        band = body.crop((0, y0, w, y1)).copy()            # size = (w, y1 - y0)
        mask = Image.new("L", band.size, 255).filter(
            ImageFilter.GaussianBlur(radius=max(0.01, feather_px / 2))
        )
        band.putalpha(mask)
        out.alpha_composite(band, (0, y0))
    return out

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

def fuse_maskblend(a: Image.Image, b: Image.Image) -> Image.Image:
    return Image.composite(a, b, radial_mask(a.size, center=(0.5,0.42), sigma=0.38))

def build_fusion(a: Image.Image, b: Image.Image, method: FusionMethod, feather_px: int) -> Image.Image:
    if method == "half":       return fuse_half(a,b,feather_px)
    if method == "headbody":   return fuse_headbody(a,b,feather_px)
    if method == "leftright":  return fuse_leftright(a,b,feather_px)
    if method == "maskblend":  return fuse_maskblend(a,b)
    return fuse_half(a,b,feather_px)

def unified_palette_color(a: Image.Image, b: Image.Image) -> Tuple[int, int, int]:
    """
    Build a cohesive target color by averaging average colors and top swatches.
    Always sanitize to RGB triples.
    """
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
def _fuse_core(spec: FuseSpec) -> Image.Image:
    a = load_sprite(spec.parents[0])
    b = load_sprite(spec.parents[1])
    fused = build_fusion(a, b, spec.method, spec.feather_px)
    if spec.harmonize:
        target = unified_palette_color(a, b)
        fused = harmonize_toward(fused, target, spec.harm_amount)
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
            import torch, torchvision.transforms as T  # optional
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

@app.get("/gallery")
def gallery(n: int = 12):
    """
    Return latest up to n cached styled/base images (data URIs) with their specs.
    Ordered by file modified time descending.
    """
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
