"""
Fine-tuning Script for RT-DETR on Construction-PPE Dataset
Using Ultralytics RT-DETR with identical splits and classes.
"""

import os
from pathlib import Path

DATA_YAML = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\construction-ppe\data_abs.yaml")
OUTPUT_DIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir\rtdetr_runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def train_rtdetr():
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")

    from ultralytics import RTDETR

    print("=" * 60)
    print("Initiating RT-DETR Fine-Tuning")
    print("=" * 60)

    # Initialize RT-DETR model (rtdetr-l)
    weights_path = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir\rtdetr-l.pt")
    model = RTDETR(str(weights_path) if weights_path.exists() else "rtdetr-l.pt")

    # Train on identical dataset split
    results = model.train(
        data=str(DATA_YAML),
        epochs=6,
        imgsz=640,
        batch=4,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(OUTPUT_DIR),
        name="rtdetr_ppe",
        amp=True,
        workers=2,
        patience=5,
        save=True
    )

    print("=" * 60)
    print("RT-DETR Training Complete!")
    print(f"Results and weights saved in: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    train_rtdetr()
