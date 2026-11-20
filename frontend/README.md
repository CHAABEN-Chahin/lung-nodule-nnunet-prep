# Quick Start — Launch Commands

## Backend (Terminal 1)

```bash
cd backend
venv\Scripts\activate
python app.py
```

Backend runs on http://localhost:5000

## Frontend (Terminal 2)

```bash
cd frontend
pnpm install
pnpm dev
```

Frontend runs on http://localhost:5173

## First-time setup (if dependencies not yet installed)

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install scikit-build-core cmake ninja setuptools-scm
pip install https://github.com/AIM-Harvard/pyradiomics/archive/refs/heads/master.zip --no-build-isolation
```

### Frontend

```bash
cd frontend
pnpm install
```

## Default credentials

- Username: `admin`
- Password: `admin`

---

# Medical Imaging Application UI

This is a code bundle for Medical Imaging Application UI. The original project is available at https://www.figma.com/design/3Hmyx1T6bTDbjCHooSCCBu/Medical-Imaging-Application-UI.
  