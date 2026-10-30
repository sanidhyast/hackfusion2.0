"""
Verification script for PPE Compliance FastAPI Backend using TestClient
Tests:
- /health endpoint
- /api/stats endpoint
- /api/detect/image endpoint with a real test image
"""

import os
import sys
import json
from pathlib import Path

# Add backend and workdir to sys.path
BACKEND_DIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\backend")
WORKDIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir")
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(WORKDIR))

from fastapi.testclient import TestClient
from app import app, load_model_on_startup

def test_backend():
    print("=" * 60)
    print("Testing PPE Compliance Detection FastAPI Backend")
    print("=" * 60)

    # 1. Trigger startup event to load model
    load_model_on_startup()

    client = TestClient(app)

    # 2. Test /health
    print("\n[1/3] Testing GET /health...")
    health_resp = client.get("/health")
    print("Health Status Code:", health_resp.status_code)
    print("Health Response:", json.dumps(health_resp.json(), indent=2))
    assert health_resp.status_code == 200

    # 3. Test /api/detect/image with test image
    test_img_path = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\construction-ppe\images\test\image1.jpeg")
    print(f"\n[2/3] Testing POST /api/detect/image with {test_img_path.name}...")
    with open(test_img_path, "rb") as f:
        files = {"file": ("image1.jpeg", f, "image/jpeg")}
        data = {"return_base64": "true", "conf_thresh": "0.35"}
        resp = client.post("/api/detect/image", files=files, data=data)

    print("Detect Status Code:", resp.status_code)
    res_json = resp.json()
    b64_preview = res_json.get("annotated_image_base64", "")[:40] + "..." if res_json.get("annotated_image_base64") else "None"
    print("Total Workers Detected:", res_json.get("total_workers"))
    print("Compliant Workers:", res_json.get("compliant_workers"))
    print("Violation Workers:", res_json.get("violation_workers"))
    print("Compliance Rate (%):", res_json.get("compliance_rate"))
    print("Latency (ms):", res_json.get("latency_ms"))
    print("Base64 Image Generated:", b64_preview)
    assert resp.status_code == 200
    assert res_json["success"] is True

    # 4. Test /api/stats
    print("\n[3/3] Testing GET /api/stats...")
    stats_resp = client.get("/api/stats")
    print("Stats Response:", json.dumps(stats_resp.json(), indent=2))
    assert stats_resp.status_code == 200

    print("\n" + "=" * 60)
    print("ALL BACKEND ENDPOINTS PASSED VERIFICATION!")
    print("=" * 60)

if __name__ == "__main__":
    test_backend()
