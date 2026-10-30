"""
PPE Compliance Backend Server Runner
Runs Uvicorn server on http://localhost:8000
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\backend")
WORKDIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir")
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(WORKDIR))

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting PPE Safety Compliance Backend...")
    print("Open Dashboard UI:  http://localhost:8000")
    print("Interactive Docs:   http://localhost:8000/docs")
    print("Health Check:       http://localhost:8000/health")
    print("=" * 60)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
