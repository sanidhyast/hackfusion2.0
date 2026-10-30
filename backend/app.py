"""
FastAPI Production Backend for PPE Compliance Detection (v1.3)
Provides RESTful APIs for image/video inspection, live metrics, and web dashboard.
"""

import os
import sys
import time
import base64
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add workdir to path
WORKDIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from association import (
    Detection,
    WorkerStatus,
    PPEAssociationEngine,
    draw_compliance_annotations
)
from infer_compliance import CompliancePipeline

# App Setup
app = FastAPI(
    title="PPE Safety Compliance AI Backend",
    description="Real-time Worker Detection, Anatomical PPE Association, and Violation Flagging API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directories
UPLOAD_DIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\backend\uploads")
PROCESSED_DIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\backend\processed")
FRONTEND_DIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\frontend")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Static mounts
app.mount("/media", StaticFiles(directory=str(PROCESSED_DIR)), name="media")
if (FRONTEND_DIR / "samples").exists():
    app.mount("/samples", StaticFiles(directory=str(FRONTEND_DIR / "samples")), name="samples")

@app.get("/", response_class=FileResponse)
def serve_dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/style.css")
def serve_css():
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")

@app.get("/app.js")
def serve_js():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")

# Global Pipeline State
PIPELINE: Optional[CompliancePipeline] = None
CHECKPOINT_PATH = WORKDIR / "rfdetr_10ep_runs" / "checkpoint_best_ema.pth"
if not CHECKPOINT_PATH.exists():
    # Fallback to 5-epoch run if 10-epoch path differs
    CHECKPOINT_PATH = WORKDIR / "rfdetr_runs" / "checkpoint_best_ema.pth"

# System Stats
STATS = {
    "total_images_processed": 0,
    "total_workers_analyzed": 0,
    "total_violations_found": 0,
    "total_compliant_workers": 0,
    "start_time": time.time()
}

@app.on_event("startup")
def load_model_on_startup():
    global PIPELINE
    print("=" * 60)
    print("Initializing PPE Compliance Detection Backend...")
    print(f"Loading Model Checkpoint: {CHECKPOINT_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Device: {device}")
    
    PIPELINE = CompliancePipeline(
        model_type="rfdetr",
        model_path=str(CHECKPOINT_PATH),
        required_ppe=["helmet", "vest"],
        conf_threshold=0.35,
        device=device
    )
    print("Backend Model Loaded Successfully & Ready for Traffic!")
    print("=" * 60)

@app.get("/health")
def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 1),
            "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 1),
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1)
        }
    
    return {
        "status": "online",
        "model_loaded": PIPELINE is not None,
        "model_type": "RF-DETR-S (10-Epoch EMA)",
        "checkpoint": str(CHECKPOINT_PATH),
        "gpu": gpu_info,
        "uptime_seconds": round(time.time() - STATS["start_time"], 1)
    }

@app.get("/api/stats")
def get_stats():
    total_w = STATS["total_workers_analyzed"]
    compliance_rate = round((STATS["total_compliant_workers"] / total_w) * 100, 2) if total_w > 0 else 100.0
    return {
        "total_images_processed": STATS["total_images_processed"],
        "total_workers_analyzed": total_w,
        "total_violations_found": STATS["total_violations_found"],
        "total_compliant_workers": STATS["total_compliant_workers"],
        "overall_compliance_rate_pct": compliance_rate
    }

class DetectResponse(BaseModel):
    success: bool
    total_workers: int
    compliant_workers: int
    violation_workers: int
    compliance_rate: float
    unassigned_items_count: int
    unassigned_items: Optional[List[Dict[str, Any]]] = None
    workers: List[Dict[str, Any]]
    latency_ms: float
    fps: float
    annotated_image_base64: Optional[str] = None
    processed_image_url: Optional[str] = None

@app.post("/api/detect/image", response_model=DetectResponse)
async def detect_image(
    file: UploadFile = File(...),
    return_base64: bool = Form(True),
    conf_thresh: float = Form(0.35)
):
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model pipeline is initializing.")

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file or unsupported image format.")

    # Process through pipeline
    annotated, summary = PIPELINE.process_image(img_bgr, conf_thresh=conf_thresh)

    # Save processed image to disk
    timestamp = int(time.time() * 1000)
    filename = f"detected_{timestamp}_{file.filename}"
    save_path = PROCESSED_DIR / filename
    cv2.imwrite(str(save_path), annotated)

    # Base64 encoding for instantaneous UI rendering
    b64_str = None
    if return_base64:
        _, buffer = cv2.imencode(".jpg", annotated)
        b64_str = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    # Update cumulative statistics
    STATS["total_images_processed"] += 1
    STATS["total_workers_analyzed"] += summary["total_workers"]
    STATS["total_violations_found"] += summary["violation_workers"]
    STATS["total_compliant_workers"] += summary["compliant_workers"]

    return {
        "success": True,
        "total_workers": summary["total_workers"],
        "compliant_workers": summary["compliant_workers"],
        "violation_workers": summary["violation_workers"],
        "compliance_rate": summary["compliance_rate"],
        "unassigned_items_count": summary["unassigned_items_count"],
        "unassigned_items": summary.get("unassigned_items", []),
        "workers": summary["workers"],
        "latency_ms": summary["latency_ms"],
        "fps": summary["fps"],
        "annotated_image_base64": b64_str,
        "processed_image_url": f"/media/{filename}"
    }

@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    max_frames: Optional[int] = Form(150),
    conf_thresh: float = Form(0.35)
):
    """Processes uploaded video frames and returns video URL and summary."""
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model pipeline is initializing.")

    # Save uploaded video
    timestamp = int(time.time() * 1000)
    in_video_path = PROCESSED_DIR / f"input_{timestamp}_{file.filename}"
    out_video_name = f"annotated_{timestamp}_{file.filename}"
    out_video_path = PROCESSED_DIR / out_video_name

    with open(in_video_path, "wb") as f:
        f.write(await file.read())

    # Process video
    frame_summaries = PIPELINE.process_video(str(in_video_path), str(out_video_path), max_frames=max_frames)

    total_frames = len(frame_summaries) if frame_summaries else 0
    total_violations = sum(f["summary"]["violation_workers"] for f in frame_summaries) if frame_summaries else 0
    total_compliant = sum(f["summary"]["compliant_workers"] for f in frame_summaries) if frame_summaries else 0
    all_workers_seen = sum(f["summary"]["total_workers"] for f in frame_summaries) if frame_summaries else 0
    peak_workers = max((f["summary"]["total_workers"] for f in frame_summaries), default=0) if frame_summaries else 0

    rate = round((total_compliant / (all_workers_seen or 1)) * 100) if all_workers_seen > 0 else 100

    # Extract distinct incident occurrences across frames
    incident_list = []
    if frame_summaries:
        for f in frame_summaries:
            frame_num = f["frame"]
            f_summary = f["summary"]
            for w in f_summary.get("workers", []):
                missing = []
                detected = w.get("detected_ppe", {})
                if not detected.get("helmet"):
                    missing.append("No helmet")
                if not detected.get("vest"):
                    missing.append("No safety vest")
                if "helmet_under_arm" in w.get("negative_flags", []) or "helmet_under_arm" in w.get("missing_ppe", []):
                    missing.append("Helmet Under Arm")
                if not detected.get("gloves") and "gloves" in detected:
                    missing.append("No gloves")

                is_comp = len(missing) == 0
                incident_list.append({
                    "frame": frame_num,
                    "timestamp_sec": round(frame_num / 25.0, 2),
                    "worker_id": w.get("worker_id", 1),
                    "confidence": round(w.get("confidence", 0.8), 2),
                    "is_compliant": is_comp,
                    "verdict": "COMPLIANT" if is_comp else "VIOLATION",
                    "missing_chips": missing if missing else ["Full Helmet & Vest"],
                    "detected_ppe": detected,
                    "negative_flags": w.get("negative_flags", []),
                    "missing_ppe": w.get("missing_ppe", [])
                })

    return {
        "success": True,
        "total_frames_processed": total_frames,
        "cumulative_violations_spotted": total_violations,
        "total_compliant_count": total_compliant,
        "peak_workers": peak_workers,
        "compliance_rate": rate,
        "video_url": f"/media/{out_video_name}",
        "raw_video_url": f"/media/input_{timestamp}_{file.filename}",
        "incidents": incident_list[:80] # Top 80 incident occurrences
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
