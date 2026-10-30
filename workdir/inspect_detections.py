import sys
from pathlib import Path
sys.path.insert(0, r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir")
from infer_compliance import CompliancePipeline
import cv2

p = CompliancePipeline()
img = cv2.imread(r"C:\Users\stpla\OneDrive\Desktop\HF2\backend\processed\detected_1793350687440_portrait-of-smiling-female-architect-holding-yellow-helmet-in-office.webp")
results = p.model.predict(img, conf=0.15)
for b in results[0].boxes:
    cid = int(b.cls[0])
    conf = float(b.conf[0])
    name = results[0].names[cid]
    xyxy = [round(float(x), 1) for x in b.xyxy[0]]
    print(f"Detected: {name} (conf={conf:.2f}) box={xyxy}")
