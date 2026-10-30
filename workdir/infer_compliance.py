"""
YOLOv8-Only PPE Compliance Detection & Violation Pipeline
Supports:
- Single image, image directories, and live video streams
- Strictly 5 detection classes:
    0: Person
    1: Helmet
    2: No Helmet
    3: Vest
    4: No Vest
- Anatomical association & per-worker compliance
- Helmet under arm violation detection
- JSON and Markdown audit report generation
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
from ultralytics import YOLO

from association import (
    Detection,
    WorkerStatus,
    PPEAssociationEngine,
    draw_compliance_annotations,
    CLASSES
)

class CompliancePipeline:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.45,
        required_ppe: Optional[List[str]] = None,
        device: str = "0"
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.required_ppe = required_ppe or ["helmet", "vest"]
        self.engine = PPEAssociationEngine(
            required_ppe=self.required_ppe,
            min_person_conf=conf_threshold,
            min_ppe_conf=max(0.15, conf_threshold - 0.10)
        )
        self.model_path = model_path or self._find_best_checkpoint()
        print(f"Initializing YOLOv8-Only Compliance Pipeline from: {self.model_path}")
        self.model = YOLO(self.model_path)

    def _find_best_checkpoint(self) -> str:
        candidates = [
            Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir\yolov8_5class_runs\yolov8s_5class_finetuned\weights\best.pt"),
            Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir\yolov8_5class_runs\yolov8s_5class_finetuned\weights\last.pt"),
            Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir\yolov8m_ppe.pt"),
            Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\yolov8s.pt")
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return "yolov8s.pt"

    def detect_and_associate(
        self,
        image: np.ndarray,
        conf: Optional[float] = None
    ) -> Tuple[List[WorkerStatus], List[Detection], Dict[str, Any], float]:
        """
        Runs YOLOv8 detection and spatial anatomical association.
        Returns:
          - worker_statuses: List[WorkerStatus]
          - unassigned_items: List[Detection]
          - summary: Dict[str, Any]
          - latency_ms: float
        """
        t0 = time.perf_counter()
        active_conf = conf if conf is not None else self.conf_threshold

        # Run YOLOv8 forward pass
        results = self.model.predict(
            source=image,
            conf=active_conf,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        t_infer = (time.perf_counter() - t0) * 1000.0

        detections = []
        if results and len(results) > 0:
            boxes_obj = results[0].boxes
            if boxes_obj is not None and len(boxes_obj) > 0:
                xyxy = boxes_obj.xyxy.cpu().numpy()
                confs = boxes_obj.conf.cpu().numpy()
                classes = boxes_obj.cls.cpu().numpy().astype(int)

                for box, score, cid in zip(xyxy, confs, classes):
                    if cid in CLASSES:
                        detections.append(
                            Detection(
                                box=[float(x) for x in box],
                                class_id=cid,
                                class_name=CLASSES[cid],
                                confidence=float(score)
                            )
                        )

        # Run association engine
        t1 = time.perf_counter()
        worker_statuses, unassigned, summary = self.engine.associate_and_evaluate(detections)
        t_assoc = (time.perf_counter() - t1) * 1000.0
        total_latency = (time.perf_counter() - t0) * 1000.0

        summary["latency_breakdown"] = {
            "yolov8_inference_ms": round(t_infer, 1),
            "association_ms": round(t_assoc, 1),
            "total_ms": round(total_latency, 1)
        }

        return worker_statuses, unassigned, summary, total_latency

    def process_image(
        self,
        image: np.ndarray,
        conf_thresh: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        workers, unassigned, summary, latency = self.detect_and_associate(image, conf=conf_thresh)
        annotated = draw_compliance_annotations(image, workers, unassigned)
        fps = round(1000.0 / latency, 1) if latency > 0 else 60.0
        summary["latency_ms"] = round(latency, 1)
        summary["fps"] = fps
        return annotated, summary

    def process_image_file(self, img_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        workers, unassigned, summary, latency = self.detect_and_associate(img)

        if output_path:
            annotated = draw_compliance_annotations(img, workers, unassigned)
            cv2.imwrite(output_path, annotated)
            summary["annotated_image_path"] = output_path

        return summary

    def process_video(
        self,
        in_video_path: str,
        out_video_path: str,
        max_frames: Optional[int] = 150,
        conf_thresh: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Processes video frames through the YOLOv8 compliance pipeline.
        Saves annotated video to out_video_path and returns per-frame summaries.
        """
        cap = cv2.VideoCapture(in_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {in_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        frame_summaries = []
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                annotated, summary = self.process_image(frame, conf_thresh=conf_thresh)
                if writer.isOpened():
                    writer.write(annotated)

                frame_summaries.append({
                    "frame": frame_idx,
                    "summary": summary
                })
        finally:
            cap.release()
            writer.release()

        return frame_summaries


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-Only PPE Compliance Detection CLI")
    parser.add_argument("--source", type=str, required=True, help="Path to image or directory")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLOv8 checkpoint")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pipeline = CompliancePipeline(model_path=args.model, conf_threshold=args.conf)

    p = Path(args.source)
    if p.is_file():
        out_img = str(Path(args.output) / f"annotated_{p.name}")
        summary = pipeline.process_image_file(str(p), out_img)
        print("\n" + "=" * 60)
        print(f"PROCESSED: {p.name} (Latency: {summary['latency_breakdown']['total_ms']} ms)")
        print(f"Workers: {summary['total_workers']} | Compliant: {summary['compliant_workers']} | Violations: {summary['violation_workers']}")
        for w in summary["workers"]:
            print(f"  Worker #{w['worker_id']}: {w['verdict']} - Missing: {w['missing_ppe']} - Flags: {w['negative_flags']}")
        print("=" * 60)

if __name__ == "__main__":
    main()
