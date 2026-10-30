"""
PPE Compliance Detection & Violation Summary CLI
Supports:
- Single images, image batches, and video files
- Configurable model backend: RF-DETR-S, RT-DETR, or YOLO
- Per-worker verdict and anatomical association
- Structured JSON and Markdown violation summaries
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from association import (
    Detection,
    WorkerStatus,
    PPEAssociationEngine,
    draw_compliance_annotations
)

CLASS_MAP = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots"
}

class CompliancePipeline:
    def __init__(
        self,
        model_type: str = "rfdetr",
        model_path: Optional[str] = None,
        required_ppe: Optional[List[str]] = None,
        conf_threshold: float = 0.35,
        device: str = "cuda"
    ):
        self.model_type = model_type.lower()
        self.device = device
        self.conf_threshold = conf_threshold
        self.required_ppe = required_ppe or ["helmet", "vest"]
        self.engine = PPEAssociationEngine(required_ppe=self.required_ppe)
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]):
        if self.model_type == "rfdetr":
            from rfdetr import RFDETRSmall
            print(f"Loading RF-DETR-S model from {model_path or 'pretrained default'}...")
            if model_path and os.path.exists(model_path):
                return RFDETRSmall.from_checkpoint(model_path)
            return RFDETRSmall()
        elif self.model_type in {"rtdetr", "yolo"}:
            from ultralytics import RTDETR, YOLO
            cls = RTDETR if self.model_type == "rtdetr" else YOLO
            path = model_path or ("rtdetr-l.pt" if self.model_type == "rtdetr" else "yolo11s.pt")
            print(f"Loading {self.model_type.upper()} model from {path}...")
            return cls(path)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def predict_image(self, img_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> Tuple[List[Detection], float]:
        """Runs model inference and returns raw normalized detections and latency (ms)."""
        t0 = time.perf_counter()
        detections = []
        thresh = conf_thresh if conf_thresh is not None else self.conf_threshold

        if self.model_type == "rfdetr":
            # RF-DETR inference on RGB image with sensitive equipment floor (0.06)
            # This ensures overhead/glared helmets and vests are captured even if person conf slider is higher
            model_thresh = min(0.06, thresh)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = self.model.predict(img_rgb, threshold=model_thresh)
            # results from supervision Detections
            if hasattr(results, "xyxy") and results.xyxy is not None and len(results.xyxy) > 0:
                for i in range(len(results.xyxy)):
                    box = results.xyxy[i].tolist()
                    cid = int(results.class_id[i]) if results.class_id is not None else 0
                    conf = float(results.confidence[i]) if results.confidence is not None else 1.0
                    cname = CLASS_MAP.get(cid, f"class_{cid}")
                    detections.append(Detection(box=box, class_id=cid, class_name=cname, confidence=conf))
        else:
            # Ultralytics RT-DETR / YOLO inference
            results = self.model.predict(img_bgr, conf=self.conf_threshold, device=self.device, verbose=False)
            res = results[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for b, c, conf in zip(boxes, clss, confs):
                cid = int(c)
                cname = res.names.get(cid, CLASS_MAP.get(cid, f"class_{cid}"))
                detections.append(Detection(box=b.tolist(), class_id=cid, class_name=cname, confidence=float(conf)))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return detections, latency_ms

    def process_image(self, img_bgr: np.ndarray, conf_thresh: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        raw_dets, latency_ms = self.predict_image(img_bgr, conf_thresh=conf_thresh)
        thresh = conf_thresh if conf_thresh is not None else self.conf_threshold
        self.engine.min_person_conf = thresh
        self.engine.min_ppe_conf = min(0.06, thresh)
        workers, unassigned, summary = self.engine.associate_and_evaluate(raw_dets)
        annotated = draw_compliance_annotations(img_bgr, workers, unassigned)
        summary["latency_ms"] = round(latency_ms, 2)
        summary["fps"] = round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0.0
        return annotated, summary

    def process_video(self, video_path: str, output_path: str, max_frames: Optional[int] = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        all_summaries = []

        print(f"Processing video: {video_path} ({total_frames} frames)...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            annotated_frame, summary = self.process_image(frame)
            out.write(annotated_frame)
            all_summaries.append({"frame": frame_idx, "summary": summary})
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames (Current FPS: {summary['fps']})")

        cap.release()
        out.release()
        print(f"Video processing finished. Saved to {output_path}")
        return all_summaries

def main():
    parser = argparse.ArgumentParser(description="PPE Compliance Detection & Association Engine")
    parser.add_argument("--source", type=str, required=True, help="Path to image, image directory, or video")
    parser.add_argument("--model-type", type=str, default="rfdetr", choices=["rfdetr", "rtdetr", "yolo"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = CompliancePipeline(
        model_type=args.model_type,
        model_path=args.model_path,
        conf_threshold=args.conf
    )

    src = Path(args.source)
    if src.is_file() and src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        out_vid = out_dir / f"annotated_{src.name}"
        pipeline.process_video(str(src), str(out_vid))
    elif src.is_file():
        img = cv2.imread(str(src))
        annotated, summary = pipeline.process_image(img)
        out_img = out_dir / f"annotated_{src.name}"
        cv2.imwrite(str(out_img), annotated)
        out_json = out_dir / f"{src.stem}_summary.json"
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Result saved to {out_img}")
        print("Violation Summary:")
        print(json.dumps(summary, indent=2))
    elif src.is_dir():
        imgs = [f for f in src.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        print(f"Processing {len(imgs)} images from {src}...")
        for p in imgs:
            img = cv2.imread(str(p))
            if img is None:
                continue
            annotated, summary = pipeline.process_image(img)
            cv2.imwrite(str(out_dir / f"annotated_{p.name}"), annotated)
        print(f"Done! Results written to {out_dir}")

if __name__ == "__main__":
    main()
