# PPE Compliance Detection - Final Comprehensive Benchmark
**Evaluated on Held-Out Test Set:** 141 images, 236 ground-truth workers, 27 unassigned/unworn items, 8 crowded scenes.

| Model / Architecture | Worker F1 (%) | Helmet F1 (%) | Vest F1 (%) | Compliance Acc (%) | Violation Recall (%) | Missed Violations | Crowded Acc (%) | Latency (ms) | FPS | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|---|
| **SH17-YOLOv8m** | 73.79% | 60.61% | 50.0% | **60.23%** | **53.93%** | 3 | 37.93% | 31.77 ms | **31.5 FPS** | 160.6 MB |
| **FineTuned-RF-DETR-S** | 85.12% | 93.09% | 86.67% | **89.81%** | **56.18%** | 13 | 89.47% | 32.68 ms | **30.6 FPS** | 308.6 MB |
| **FineTuned-RT-DETR** | 74.62% | 86.42% | 83.25% | **89.69%** | **41.57%** | 18 | 76.67% | 49.34 ms | **20.3 FPS** | 464.6 MB |
| **Ensemble: RF-DETR-S + SH17 (WBF)** | 82.2% | 90.95% | 72.85% | **86.6%** | **56.18%** | 9 | 78.12% | 58.82 ms | **17.0 FPS** | 440.7 MB |
| **Ensemble: RF-DETR-S + RT-DETR (WBF)** | 84.37% | 89.88% | 84.4% | **91.88%** | **53.93%** | 12 | 87.88% | 81.46 ms | **12.3 FPS** | 464.6 MB |
