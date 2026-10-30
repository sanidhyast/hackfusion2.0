"""
YOLO to COCO Conversion Script for Construction-PPE Dataset
Generates train.json, val.json, and test.json in standard COCO format.
"""

import os
import json
from pathlib import Path
from PIL import Image

DATASET_ROOT = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\construction-ppe")

CATEGORIES = [
    {"id": 0, "name": "helmet", "supercategory": "ppe"},
    {"id": 1, "name": "gloves", "supercategory": "ppe"},
    {"id": 2, "name": "vest", "supercategory": "ppe"},
    {"id": 3, "name": "boots", "supercategory": "ppe"},
    {"id": 4, "name": "goggles", "supercategory": "ppe"},
    {"id": 5, "name": "none", "supercategory": "none"},
    {"id": 6, "name": "Person", "supercategory": "human"},
    {"id": 7, "name": "no_helmet", "supercategory": "violation"},
    {"id": 8, "name": "no_goggle", "supercategory": "violation"},
    {"id": 9, "name": "no_gloves", "supercategory": "violation"},
    {"id": 10, "name": "no_boots", "supercategory": "violation"},
]

def convert_split(split_name: str, images_dir: Path, labels_dir: Path, out_json_path: Path):
    print(f"Converting split: {split_name}...")
    coco_data = {
        "info": {
            "description": f"Construction-PPE {split_name} split in COCO format",
            "version": "1.0",
            "year": 2026
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions])
    
    annotation_id = 1
    
    for img_id, img_path in enumerate(image_files, 1):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception as e:
            print(f"Skipping unreadable image {img_path}: {e}")
            continue

        coco_data["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })

        # Look for corresponding label file
        label_file = labels_dir / f"{img_path.stem}.txt"
        if not label_file.exists():
            continue

        with open(label_file, "r") as lf:
            for line in lf:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc = float(parts[1]) * w
                yc = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h

                # Convert from YOLO center (xc, yc, w, h) to COCO top-left (x1, y1, w, h)
                x1 = max(0.0, xc - bw / 2.0)
                y1 = max(0.0, yc - bh / 2.0)
                bw = min(bw, w - x1)
                bh = min(bh, h - y1)
                area = bw * bh

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                    "area": round(area, 2),
                    "iscrowd": 0
                })
                annotation_id += 1

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as out_f:
        json.dump(coco_data, out_f)
    print(f"Saved {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations to {out_json_path}")

def main():
    annotations_dir = DATASET_ROOT / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train", DATASET_ROOT / "images" / "train", DATASET_ROOT / "labels" / "train"),
        ("val", DATASET_ROOT / "images" / "val", DATASET_ROOT / "labels" / "val"),
        ("test", DATASET_ROOT / "images" / "test", DATASET_ROOT / "labels" / "test")
    ]

    for name, img_dir, lbl_dir in splits:
        out_json = annotations_dir / f"{name}.json"
        convert_split(name, img_dir, lbl_dir, out_json)

if __name__ == "__main__":
    main()
