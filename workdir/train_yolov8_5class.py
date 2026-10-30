"""
Train and Fine-Tune YOLOv8 Strictly on 5-Class Unified Industrial PPE Dataset
Classes:
  0: Person
  1: Helmet
  2: No Helmet
  3: Vest
  4: No Vest

Evaluates on held-out test split and logs class-wise precision, recall, mAP@50, mAP@50-95.
"""

import os
import json
import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    base_dir = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2")
    data_yaml = base_dir / "datasets" / "unified_ppe_5class" / "dataset.yaml"
    runs_dir = base_dir / "workdir" / "yolov8_5class_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("STARTING YOLOV8-ONLY 5-CLASS FINE-TUNING")
    print(f"Dataset YAML: {data_yaml}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 70)

    # 1. Initialize YOLOv8s pretrained weights
    model = YOLO("yolov8s.pt")
    
    # 2. Train on balanced 5-class industrial dataset
    # 15 epochs provides solid fine-tuning convergence on 3,000 images without overfitting
    train_results = model.train(
        data=str(data_yaml),
        epochs=15,
        imgsz=640,
        batch=16,
        workers=2,
        device=0,
        project=str(runs_dir),
        name="yolov8s_5class_finetuned",
        exist_ok=True,
        pretrained=True,
        amp=True,
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE! NOW RUNNING RIGOROUS TEST SET EVALUATION")
    print("=" * 70)
    
    best_weights = runs_dir / "yolov8s_5class_finetuned" / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = runs_dir / "yolov8s_5class_finetuned" / "weights" / "last.pt"
    
    eval_model = YOLO(str(best_weights))
    
    # Evaluate on held-out test split
    metrics = eval_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        project=str(runs_dir),
        name="test_evaluation",
        exist_ok=True
    )
    
    print("\n" + "=" * 70)
    print("CLASS-WISE PERFORMANCE REPORT ON TEST SET")
    print("=" * 70)
    
    class_names = ["Person", "Helmet", "No Helmet", "Vest", "No Vest"]
    
    # Extract class-wise metrics
    p_per_class = metrics.box.p
    r_per_class = metrics.box.r
    map50_per_class = metrics.box.ap50
    map_per_class = metrics.box.ap
    
    print(f"{'Class ID':<10} {'Class Name':<15} {'Precision':<12} {'Recall':<12} {'mAP@50':<12} {'mAP@50-95':<12}")
    print("-" * 75)
    
    summary_data = {
        "model": "YOLOv8s-Industrial-5Class",
        "checkpoint": str(best_weights),
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map)
        },
        "classes": {}
    }
    
    for i, name in enumerate(class_names):
        p = float(p_per_class[i]) if i < len(p_per_class) else 0.0
        r = float(r_per_class[i]) if i < len(r_per_class) else 0.0
        m50 = float(map50_per_class[i]) if i < len(map50_per_class) else 0.0
        m = float(map_per_class[i]) if i < len(map_per_class) else 0.0
        
        summary_data["classes"][name] = {
            "class_id": i,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "map50": round(m50, 4),
            "map50_95": round(m, 4)
        }
        print(f"{i:<10} {name:<15} {p:<12.4f} {r:<12.4f} {m50:<12.4f} {m:<12.4f}")
        
    print("-" * 75)
    print(f"{'ALL':<10} {'Overall':<15} {metrics.box.mp:<12.4f} {metrics.box.mr:<12.4f} {metrics.box.map50:<12.4f} {metrics.box.map:<12.4f}")
    print("=" * 75)
    
    # Save summary to JSON
    json_path = runs_dir / "metrics_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved full performance report to: {json_path}")

if __name__ == "__main__":
    main()
