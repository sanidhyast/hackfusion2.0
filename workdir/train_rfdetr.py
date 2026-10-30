"""
Fine-tuning Script for RF-DETR-S on Construction-PPE Dataset
"""

import os
import sys
import io
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "0"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

DATASET_DIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\construction-ppe")
OUTPUT_DIR = Path(r"C:\Users\stpla\OneDrive\Desktop\HF2\workdir\rfdetr_runs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def train_rfdetr():
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")

    from rfdetr import RFDETRSmall

    print("=" * 60)
    print("Initiating RF-DETR-S Fine-Tuning")
    print("=" * 60)

    model = RFDETRSmall()

    # Training configuration optimized for 6GB RTX 3050 GPU
    model.train(
        dataset_dir=str(DATASET_DIR),
        epochs=8,
        batch_size=4,
        grad_accum_steps=4,  # Effective batch size = 16
        lr=1e-4,
        output_dir=str(OUTPUT_DIR),
        num_workers=2,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("RF-DETR-S Training Complete!")
    print(f"Checkpoints saved in: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    train_rfdetr()
