# AAC (Accelerated Arbitrage Corporation)

This repository contains the source code for AAC system.  The history has been cleaned of large data blobs and extraneous assets.

## Setup

```powershell
cd <repo-root>
python -m venv .venv
.\.venv\Scripts\activate   # windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running

Launch the main application (example):

```powershell
python -m aac    # or appropriate entrypoint
```

### Notes

- The `requirements.txt` lists all installed dependencies.
- Source packages are under `/src` (to be reorganized as needed).
- All generated data and large assets are ignored via `.gitignore`.
