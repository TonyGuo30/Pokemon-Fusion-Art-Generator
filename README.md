# Pokémon Fusion Art

This repo provides a minimal, **fully working** pipeline with:
- **Fusion:** pixel fusion with simple part masks and palette harmonization (`POST /fuse`).
- **Style Transfer:** stylized output via `POST /style` (tries PyTorch if installed, otherwise filter-based).
- **Integration + Web:** a Vite + React UI to run both and preview results.

## Run

### Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```

### Frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```

## Customize
- Replace images in `backend/assets/` with your own sprites (same 64×64 transparent PNGs).
- Extend `/style` to a full neural style transfer (AdaIN, fast style transfer) if you install `torch` + `torchvision`.
