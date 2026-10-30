# PPE Compliance AI Platform (HackFusion 2.0)

An enterprise-grade, real-time Computer Vision and Deep Learning PPE (Personal Protective Equipment) compliance monitoring platform powered by fine-tuned **RF-DETR-S (DINOv2 backbone)** with Hungarian Matching anatomical spatial association.

---

## Key Features

- **4-Screen CCTV Live Monitoring Console:**
  - Multi-camera stream ingestion (webcam / IP camera / RTSP).
  - **Per-Camera Room Names & Enforcement Rules:** Assign custom location tags (e.g. *Welding Bay*, *Main Scaffold*) and select specific PPE checks enforced only for that feed.
  - **Single-Focus Double-Click Layout:** Double-click any camera to expand it into a large focused display on the left with the other feeds stacked vertically on the right.
- **Deep Spatial PPE Association Engine:**
  - Hungarian algorithm bipartite matching associating PPE elements with anatomical human zones (head, torso, arms/hands, feet, eyes).
  - Identifies compliance breaches including **No Helmet**, **No Safety Vest**, **Helmet Under Arm (Carried Helmet)**, **No Gloves**, **No Boots**, and **No Eye Protection**.
- **Static Image & Forensic Video Auditing:**
  - High-resolution photo inspection with coordinate overlays, confidence thresholds, and per-worker breakdown.
  - Frame-by-frame video timeline audit with incident timestamps.
- **Enterprise Compliance Audit & Visual Proofs:**
  - Audit logging with automated visual snapshot proof capture compositing video/image frames with AI bounding boxes.
  - Interactive proof lightbox modal with high-res inspection and evidence download.
  - Export audit logs to CSV or JSON.

---

## Project Structure

```
├── backend/
│   ├── app.py              # FastAPI detection server & API endpoints
│   ├── run_server.py       # Uvicorn server launcher
│   └── test_api.py         # Backend health and endpoint tests
├── frontend/
│   ├── index.html          # Dark-mode Operations Console
│   ├── style.css           # Styling with responsive layouts
│   ├── app.js              # State controller & Canvas drawing engine
│   └── samples/            # Test images
├── workdir/
│   ├── association.py      # Spatial & anatomical PPE-to-worker matching
│   ├── infer_compliance.py # Model inference pipeline
│   └── ...                 # Model training & evaluation scripts
└── README.md
```

---

## Getting Started

### 1. Requirements & Dependencies
Ensure Python 3.10+ and PyTorch with CUDA support are installed:
```bash
pip install fastapi uvicorn torch torchvision opencv-python Pillow pyyaml numpy scipy
```

### 2. Running the Backend
From the root directory or `backend/`:
```bash
python backend/run_server.py
```
Backend API runs at `http://localhost:8000`.

### 3. Running the Frontend
Open `frontend/index.html` in your browser or serve it through the FastAPI backend:
```
http://localhost:8000
```
