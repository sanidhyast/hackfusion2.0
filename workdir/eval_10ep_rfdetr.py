"""
Evaluation of the 10-Epoch RF-DETR-S Checkpoint on Held-out Test Set
"""

import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch

# Ensure workdir is on path
WORKDIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir")
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from association import Detection, WorkerStatus, PPEAssociationEngine
from evaluate_and_compare import evaluate_model_pipeline, CLASS_MAP, TEST_IMAGES_DIR

def main():
    print("=" * 60)
    print("Evaluating New 10-Epoch RF-DETR-S on Held-Out Test Set")
    print("=" * 60)

    checkpoint_path = WORKDIR / "rfdetr_10ep_runs" / "checkpoint_best_ema.pth"
    if not checkpoint_path.exists():
        print(f"Error: checkpoint {checkpoint_path} not found.")
        return

    from rfdetr import RFDETRSmall
    model = RFDETRSmall.from_checkpoint(str(checkpoint_path))

    def predict_rfdetr_10ep(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = model.predict(img_rgb, threshold=0.35)
        dets = []
        if hasattr(res, "xyxy") and res.xyxy is not None and len(res.xyxy) > 0:
            for i in range(len(res.xyxy)):
                box = res.xyxy[i].tolist()
                cid = int(res.class_id[i]) if res.class_id is not None else 0
                conf = float(res.confidence[i]) if res.confidence is not None else 1.0
                cname = CLASS_MAP.get(cid, f"class_{cid}")
                dets.append(Detection(box=box, class_id=cid, class_name=cname, confidence=conf))
        return dets

    engine = PPEAssociationEngine(required_ppe=["helmet", "vest"])
    test_images = sorted([p for p in TEST_IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    metrics = evaluate_model_pipeline(
        model_name="FineTuned-RF-DETR-S (10 Epochs)",
        predict_fn=predict_rfdetr_10ep,
        engine=engine,
        test_images=test_images,
        save_vis=False
    )

    print("\n" + "=" * 60)
    print("COMPARISON: 5 Epochs vs 10 Epochs")
    print("=" * 60)
    print(f"Worker F1:                     85.12% -> {metrics['worker_f1_pct']:.2f}%")
    print(f"Helmet F1:                     93.09% -> {metrics['helmet_f1_pct']:.2f}%")
    print(f"Vest F1:                       86.67% -> {metrics['vest_f1_pct']:.2f}%")
    print(f"Compliance Verdict Accuracy:   89.81% -> {metrics['compliance_accuracy_pct']:.2f}%")
    print(f"Missed Violations:             13     -> {metrics['missed_violations']}")
    print(f"False Alarm Violations:        8      -> {metrics['false_alarm_violations']}")
    print(f"Crowded Scene Accuracy:        89.47% -> {metrics['crowded_scene_accuracy_pct']:.2f}%")
    print(f"Speed:                         30.6 FPS -> {metrics['fps']:.1f} FPS")
    print("=" * 60)

    # Save JSON summary
    out_file = WORKDIR / "rfdetr_10ep_runs" / "test_evaluation_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {out_file}")

if __name__ == "__main__":
    main()
