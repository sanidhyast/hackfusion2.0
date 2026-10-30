"""
Unified 5-Class Industrial PPE Dataset Builder & Balancer
Combines CSS, Construction-PPE, and Constructionk8mm datasets.
Maps strictly to:
  0: Person
  1: Helmet
  2: No Helmet
  3: Vest
  4: No Vest
Filters all other classes.
Applies controlled class-balancing to training split (within ~10% for PPE classes).
Maintains natural, realistic distributions for val and test splits (no artificial test set manipulation).
Generates:
  datasets/unified_ppe_5class/
    images/{train, val, test}
    labels/{train, val, test}
    dataset.yaml
"""

import os
import shutil
import random
import json
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image

random.seed(42)

BASE_DIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2")
OUT_DIR = BASE_DIR / "datasets" / "unified_ppe_5class"

CSS_DIR = BASE_DIR / "datasets" / "css" / "css-data"
CONSTR_PPE_DIR = BASE_DIR / "construction-ppe"
K8MM_DIR = BASE_DIR / "datasets" / "constructionk8mm"

CLASS_NAMES = ["Person", "Helmet", "No Helmet", "Vest", "No Vest"]

def clean_and_prepare_dirs():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def load_css_data():
    """
    CSS dataset mapping:
      0: Hardhat        -> 1 (Helmet)
      2: NO-Hardhat     -> 2 (No Helmet)
      4: NO-Safety Vest -> 4 (No Vest)
      5: Person         -> 0 (Person)
      7: Safety Vest    -> 3 (Vest)
      Others (Mask, Cone, Machinery, Vehicle) -> Dropped
    """
    mapping = {0: 1, 2: 2, 4: 4, 5: 0, 7: 3}
    items = []
    for split_dir in ["train", "valid", "test"]:
        img_dir = CSS_DIR / split_dir / "images"
        lbl_dir = CSS_DIR / split_dir / "labels"
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if not img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            boxes = []
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig_cid = int(parts[0])
                        if orig_cid in mapping:
                            target_cid = mapping[orig_cid]
                            coords = [float(x) for x in parts[1:5]]
                            boxes.append((target_cid, coords))
            if boxes:
                items.append({
                    "source": "CSS",
                    "img_path": img_path,
                    "boxes": boxes,
                    "stem": f"css_{img_path.stem}"
                })
    print(f"Loaded {len(items)} usable images from CSS")
    return items

def load_cppe_data():
    """
    Construction-PPE mapping:
      0: helmet         -> 1 (Helmet)
      2: vest           -> 3 (Vest)
      6: Person         -> 0 (Person)
      7: no_helmet      -> 2 (No Helmet)
      Others (gloves, boots, goggles, none) -> Dropped
    """
    mapping = {0: 1, 2: 3, 6: 0, 7: 2}
    items = []
    for split_dir in ["train", "val", "test"]:
        img_dir = CONSTR_PPE_DIR / "images" / split_dir
        lbl_dir = CONSTR_PPE_DIR / "labels" / split_dir
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if not img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            boxes = []
            with open(lbl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        orig_cid = int(parts[0])
                        if orig_cid in mapping:
                            target_cid = mapping[orig_cid]
                            coords = [float(x) for x in parts[1:5]]
                            boxes.append((target_cid, coords))
            if boxes:
                items.append({
                    "source": "CPPE",
                    "img_path": img_path,
                    "boxes": boxes,
                    "stem": f"cppe_{img_path.stem}"
                })
    print(f"Loaded {len(items)} usable images from Construction-PPE")
    return items

def load_k8mm_data():
    """
    Constructionk8mm (COCO format) mapping:
      2: helmet, 3: helmet on -> 1 (Helmet)
      6: no helmet            -> 2 (No Helmet)
      7: no vest              -> 4 (No Vest)
      8: person               -> 0 (Person)
      9: vest                 -> 3 (Vest)
    """
    mapping = {2: 1, 3: 1, 6: 2, 7: 4, 8: 0, 9: 3}
    items = []
    for split_dir in ["train", "test", "valid-mini"]:
        folder = K8MM_DIR / split_dir
        coco_file = folder / "_annotations.coco.json"
        if not coco_file.exists():
            continue
        with open(coco_file, "r") as f:
            coco = json.load(f)
        
        img_id_to_file = {}
        img_id_to_size = {}
        for img_info in coco.get("images", []):
            img_id_to_file[img_info["id"]] = img_info["file_name"]
            img_id_to_size[img_info["id"]] = (img_info["width"], img_info["height"])
        
        anns_by_img = defaultdict(list)
        for ann in coco.get("annotations", []):
            cid = ann["category_id"]
            if cid in mapping:
                target_cid = mapping[cid]
                anns_by_img[ann["image_id"]].append((target_cid, ann["bbox"]))
        
        for img_id, anns in anns_by_img.items():
            fname = img_id_to_file.get(img_id)
            if not fname:
                continue
            img_path = folder / fname
            if not img_path.exists():
                continue
            
            w_img, h_img = img_id_to_size[img_id]
            if w_img <= 0 or h_img <= 0:
                continue
            
            boxes = []
            for target_cid, (x, y, w, h) in anns:
                # Convert COCO (x_min, y_min, w, h) to YOLO (cx, cy, w, h) normalized
                cx = (x + w / 2.0) / w_img
                cy = (y + h / 2.0) / h_img
                nw = w / w_img
                nh = h / h_img
                # Clip to valid [0, 1] range
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.001, min(1.0, nw))
                nh = max(0.001, min(1.0, nh))
                boxes.append((target_cid, [cx, cy, nw, nh]))
            
            if boxes:
                items.append({
                    "source": "K8MM",
                    "img_path": img_path,
                    "boxes": boxes,
                    "stem": f"k8mm_{img_path.stem}"
                })
    print(f"Loaded {len(items)} usable images from Constructionk8mm")
    return items

def balance_and_split(all_items):
    """
    1. Stratified split: 80% train_pool, 10% val, 10% test.
    2. Hold out val and test WITHOUT artificial balancing to reflect real-world distributions.
    3. Balance train_pool across all 5 classes within ~10% representation.
    """
    random.shuffle(all_items)
    n = len(all_items)
    n_test = int(0.10 * n)
    n_val = int(0.10 * n)
    
    test_items = all_items[:n_test]
    val_items = all_items[n_test:n_test + n_val]
    train_pool = all_items[n_test + n_val:]
    
    print(f"Pool Split: TrainPool={len(train_pool)}, Val={len(val_items)}, Test={len(test_items)}")
    
    # Analyze train pool class counts
    pool_counts = Counter()
    for it in train_pool:
        for c, _ in it["boxes"]:
            pool_counts[c] += 1
    
    print(f"Raw Train Pool Counts: {dict(pool_counts)}")
    # Class 2 (No Helmet) and Class 4 (No Vest) are minority classes
    target_count = min(pool_counts[2], pool_counts[4])
    # Target balancing per class around min instance count
    print(f"Target count per class for training: ~{target_count}")
    
    # Sort train pool: prioritize images with minority classes (2 and 4)
    # and fewer excess person boxes
    def item_score(it):
        classes = [c for c, _ in it["boxes"]]
        has_2 = 2 in classes
        has_4 = 4 in classes
        num_person = sum(1 for c in classes if c == 0)
        # Higher score = higher priority
        priority = 0
        if has_2 and has_4:
            priority = 100
        elif has_2:
            priority = 80
        elif has_4:
            priority = 60
        elif 3 in classes:
            priority = 40
        elif 1 in classes:
            priority = 20
        # Penalize crowd images with > 4 persons to keep person count balanced
        return priority - (num_person * 2)

    train_pool.sort(key=item_score, reverse=True)
    
    selected_train = []
    train_counts = Counter()
    
    # Target range: within 10% of target
    max_cap = int(target_count * 1.15)
    
    for it in train_pool:
        it_classes = [c for c, _ in it["boxes"]]
        # Check if adding this image helps reach target without exceeding max_cap
        needs_classes = any(train_counts[c] < target_count for c in it_classes if c > 0)
        
        # Always add if contains under-represented minority class
        if train_counts[2] < target_count and 2 in it_classes:
            selected_train.append(it)
            for c, _ in it["boxes"]:
                train_counts[c] += 1
        elif train_counts[4] < target_count and 4 in it_classes:
            selected_train.append(it)
            for c, _ in it["boxes"]:
                train_counts[c] += 1
        elif (train_counts[1] < target_count and 1 in it_classes) or (train_counts[3] < target_count and 3 in it_classes):
            selected_train.append(it)
            for c, _ in it["boxes"]:
                train_counts[c] += 1
        elif all(train_counts[c] < max_cap for c in it_classes if c > 0):
            if train_counts[0] < max_cap * 1.5:
                selected_train.append(it)
                for c, _ in it["boxes"]:
                    train_counts[c] += 1

    return selected_train, val_items, test_items

def write_split(split_name, items):
    img_out = OUT_DIR / "images" / split_name
    lbl_out = OUT_DIR / "labels" / split_name
    
    counts = Counter()
    for idx, it in enumerate(items):
        ext = it["img_path"].suffix.lower()
        new_name = f"{split_name}_{it['stem']}_{idx:05d}{ext}"
        lbl_name = f"{split_name}_{it['stem']}_{idx:05d}.txt"
        
        # Copy image
        shutil.copy2(it["img_path"], img_out / new_name)
        
        # Write YOLO label file
        with open(lbl_out / lbl_name, "w") as f:
            for cid, coords in it["boxes"]:
                counts[cid] += 1
                coord_str = " ".join(f"{x:.6f}" for x in coords)
                f.write(f"{cid} {coord_str}\n")
                
    return counts

def main():
    print("=" * 70)
    print("STEP 1: INITIALIZING UNIFIED 5-CLASS DATASET DIRECTORIES")
    print("=" * 70)
    clean_and_prepare_dirs()
    
    print("\nSTEP 2: LOADING AND NORMALIZING ANNOTATIONS")
    all_items = []
    all_items.extend(load_css_data())
    all_items.extend(load_cppe_data())
    all_items.extend(load_k8mm_data())
    print(f"Total raw candidate images across all sources: {len(all_items)}")
    
    print("\nSTEP 3: BALANCING AND SPLITTING DATASET")
    train_items, val_items, test_items = balance_and_split(all_items)
    
    print("\nSTEP 4: WRITING DATASET FILES")
    train_counts = write_split("train", train_items)
    val_counts = write_split("val", val_items)
    test_counts = write_split("test", test_items)
    
    print("\nSTEP 5: GENERATING DATASET YAML")
    yaml_content = f"""path: {OUT_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 5
names: {CLASS_NAMES}
"""
    yaml_path = OUT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Written dataset config to: {yaml_path}")
    
    print("\n" + "=" * 70)
    print("DATASET BALANCING & AUDIT SUMMARY REPORT")
    print("=" * 70)
    print(f"{'Class ID':<10} {'Class Name':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 70)
    for cid in range(5):
        cname = CLASS_NAMES[cid]
        tr = train_counts[cid]
        va = val_counts[cid]
        te = test_counts[cid]
        tot = tr + va + te
        print(f"{cid:<10} {cname:<15} {tr:<10} {va:<10} {te:<10} {tot:<10}")
    print("-" * 70)
    print(f"{'Total Imgs':<10} {'':<15} {len(train_items):<10} {len(val_items):<10} {len(test_items):<10} {len(train_items)+len(val_items)+len(test_items):<10}")
    print("=" * 70)

if __name__ == "__main__":
    main()
