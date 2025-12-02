# Pokémon Fusion Art — Simple All-Functions Demo

This repo provides a minimal, **fully working** pipeline with:
- **Fusion:** pixel fusion with simple part masks and palette harmonization (`POST /fuse`).
- **Style Transfer:** stylized output via `POST /style` (tries PyTorch if installed, otherwise filter-based).
- **Integration + Web:** a Vite + React UI to run both and preview results.

## Directory Structure

```bash
├── backend : backend engine
│   ├── GAN : AI training module based on CycleGAN to perform unpaired Image-to-Image translation
│   ├── VGG : Neural Style Transfer using a pre-trained VGG19 network
│   └── fusion : Image merging/fustion using Hugging Faces pretrained diffusion models 
└── frontend
    ├── node_modules
    └── src
```


## Run

### Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
# http://127.0.0.1:8000/health
```

### Fusion Web Demo
```bash
python backend/fusion/fusion_server.py
# Then Use web browser to Open frontend/fusion.html
```


### Frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
# open http://127.0.0.1:5173
```

## Customize
- Replace images in `backend/assets/` with your own sprites (same 64×64 transparent PNGs).
- Extend `/style` to a full neural style transfer (AdaIN, fast style transfer) if you install `torch` + `torchvision`.
