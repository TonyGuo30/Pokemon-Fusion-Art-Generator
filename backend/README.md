# Backend — Fusion Art Simple (FastAPI)

Endpoints
- `GET /health`
- `GET /creatures`
- `POST /fuse` → pixel fusion with optional palette harmonization (returns base64 PNG)
- `POST /style` → runs fusion then applies a styled filter; if PyTorch is installed, uses a lightweight tensor pass

Run
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload
```
