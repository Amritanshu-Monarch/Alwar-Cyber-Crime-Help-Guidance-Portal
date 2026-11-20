

# Run the Project

These commands are for Windows PowerShell.

## 1. Requirements

Install:

- Python 3.11, 3.12, or 3.13
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended)
- Node.js 18+ (only needed for the optional static server)
- A Google Gemini API key for chat and voice features

## 2. Configure the backend

From the project root:

```powershell
cd backend
Copy-Item .env.example .env
notepad .env
```

Set `GOOGLE_API_KEY` in `backend/.env`. Do not commit this file.

## 3. Install dependencies

```powershell
cd backend
uv sync
```

If the project is in a cloud-synced Windows folder, the included `backend/uv.toml`
automatically makes uv copy packages instead of creating unsupported hardlinks.
If an existing `.venv` was created in another folder, regenerate its launchers once:

```powershell
uv sync --reinstall
```

If `uv` is not installed, use Python instead:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run PowerShell as your user and execute:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 4. Start the application

### Recommended: backend serves the website and API

Open a terminal in the project root:

```powershell
cd backend
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>.

The knowledge base is indexed automatically on the first startup. To index it manually:

```powershell
cd backend
uv run python -m app.services.rag
```

### Optional: run the Node.js static server

Use this when you want the static website on port `3000`. Keep the backend running in another terminal:

```powershell
cd backend
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd ..
$env:BACKEND_PORT = "8000"
node server.js
```

Open <http://localhost:3000>. The Node server proxies `/api/*` requests to the backend.

## 5. Verify the backend

With the backend running:

```powershell
Invoke-RestMethod http://localhost:8000/api/health
```

Expected response:

```text
ok
--
True
```

## 6. Run tests

```powershell
cd backend
uv run pytest
```

Tests that call Gemini are marked as integration tests and are skipped by default. A valid `GOOGLE_API_KEY` is required for those tests.

## Common commands

```powershell
# Stop a running server
Ctrl+C

# Use a different backend port
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8001

# Use a different Node/static-server port
$env:PORT = "3001"
node server.js
```
