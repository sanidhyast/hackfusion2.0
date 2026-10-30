"""
Test Dataset Distribution Analyzer
Analyzes ground truth distribution, worker counts, compliance status, and unassigned items.
"""

from pathlib import Path
from collections import Counter
from association import PPEAssociationEngine, Detection

DATASET_ROOT = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\construction-ppe")
TEST_IMAGES_DIR = DATASET_ROOT / "images" / "test"
TEST_LABELS_DIR = DATASET_ROOT / "labels" / "test"

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

def analyze():
    engine = PPEAssociationEngine(required_ppe=["helmet", "vest"])
    
    total_images = 0
    worker_count_hist = Counter()
    total_workers = 0
    compliant_workers = 0
    violation_workers = 0
    missing_items_hist = Counter()
    unassigned_equipment_hist = Counter()
    crowded_scenes = 0  # >= 4 workers

    for label_file in sorted(TEST_LABELS_DIR.glob("*.txt")):
        total_images += 1
        dets = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                xc, yc, w, h = [float(v) for v in parts[1:5]]
                # Normalized box
                x1 = xc - w / 2.0
                y1 = yc - h / 2.0
                x2 = xc + w / 2.0
                y2 = yc + h / 2.0
                dets.append(Detection(
                    box=[x1, y1, x2, y2],
                    class_id=cid,
                    class_name=CLASS_MAP.get(cid, str(cid)),
                    confidence=1.0
                ))

        workers, unassigned, summary = engine.associate_and_evaluate(dets)
        n_workers = len(workers)
        worker_count_hist[n_workers] += 1
        total_workers += n_workers
        compliant_workers += summary["compliant_workers"]
        violation_workers += summary["violation_workers"]

        if n_workers >= 4:
            crowded_scenes += 1

        for w in workers:
            for m in w.missing_ppe:
                missing_items_hist[m] += 1

        for u in unassigned:
            unassigned_equipment_hist[u.class_name] += 1

    print("=" * 60)
    print("CONSTRUCTION-PPE TEST SPLIT ANALYSIS (GROUND TRUTH)")
    print("=" * 60)
    print(f"Total Test Images: {total_images}")
    print(f"Total Workers in Test Set: {total_workers}")
    print(f"Compliant Workers: {compliant_workers} ({compliant_workers/total_workers*100:.1f}%)")
    print(f"Violation Workers: {violation_workers} ({violation_workers/total_workers*100:.1f}%)")
    print(f"Crowded Scenes (>= 4 workers): {crowded_scenes} images")
    print("\nWorker Count per Image Distribution:")
    for k in sorted(worker_count_hist.keys()):
        print(f"  {k} workers: {worker_count_hist[k]} images")
    print("\nMissing PPE Breakdown in Ground Truth:")
    for k, v in missing_items_hist.most_common():
        print(f"  {k}: {v}")
    print("\nUnassigned / Loose Equipment (Held in hand, on table, background):")
    for k, v in unassigned_equipment_hist.most_common():
        print(f"  {k}: {v}")
    print("=" * 60)

if __name__ == "__main__":
    analyze()
