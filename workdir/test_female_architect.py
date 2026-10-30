import sys
from pathlib import Path
sys.path.insert(0, r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir")
from infer_compliance import CompliancePipeline
import cv2

p = CompliancePipeline()
img_path = r"C:\Users\stpla\OneDrive\Desktop\HF2\backend\processed\detected_1793350687440_portrait-of-smiling-female-architect-holding-yellow-helmet-in-office.webp"
annotated, summary = p.process_image(cv2.imread(img_path))

print("Total Workers:", summary["total_workers"])
print("Compliant Workers:", summary["compliant_workers"])
print("Violation Workers:", summary["violation_workers"])
print("Unassigned items count:", summary["unassigned_items_count"])
for w in summary["workers"]:
    print(f"Worker #{w['worker_id']}: {w['verdict']} | missing={w['missing_ppe']} | flags={w['negative_flags']}")
