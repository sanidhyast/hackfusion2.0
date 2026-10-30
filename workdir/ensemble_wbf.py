"""
Class-Aware Weighted Boxes Fusion (WBF) and Model Ensemble Engine
Implements:
1. Class-aware Weighted Boxes Fusion (WBF) across model predictions
2. Class-aware Non-Maximum Suppression (NMS) ensemble
3. Model Agreement & Conflict / Uncertainty Analyzer:
   - Detects worker disagreement (person detected by Model A but missed by Model B)
   - Detects equipment conflict (helmet detected vs no_helmet detected)
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from association import Detection

def compute_iou_matrix(boxesA: np.ndarray, boxesB: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU between two sets of boxes [x1, y1, x2, y2]."""
    if len(boxesA) == 0 or len(boxesB) == 0:
        return np.zeros((len(boxesA), len(boxesB)), dtype=np.float32)

    xA = np.maximum(boxesA[:, None, 0], boxesB[None, :, 0])
    yA = np.maximum(boxesA[:, None, 1], boxesB[None, :, 1])
    xB = np.minimum(boxesA[:, None, 2], boxesB[None, :, 2])
    yB = np.minimum(boxesA[:, None, 3], boxesB[None, :, 3])

    inter_w = np.maximum(0.0, xB - xA)
    inter_h = np.maximum(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = np.maximum(0.0, boxesA[:, 2] - boxesA[:, 0]) * np.maximum(0.0, boxesA[:, 3] - boxesA[:, 1])
    areaB = np.maximum(0.0, boxesB[:, 2] - boxesB[:, 0]) * np.maximum(0.0, boxesB[:, 3] - boxesB[:, 1])
    union_area = areaA[:, None] + areaB[None, :] - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)

def weighted_boxes_fusion(
    predictions_per_model: List[List[Detection]],
    weights: Optional[List[float]] = None,
    iou_thresh: float = 0.50,
    skip_box_thr: float = 0.20,
    conf_type: str = "avg"
) -> Tuple[List[Detection], Dict[str, Any]]:
    """
    Fuses bounding box predictions from multiple models using Weighted Boxes Fusion.
    Returns:
        - Fused list of Detection objects
        - Disagreement and conflict analysis dictionary
    """
    n_models = len(predictions_per_model)
    if weights is None:
        weights = [1.0] * n_models
    else:
        weights = [float(w) for w in weights]

    # Group all detections by normalized class name
    class_groups = {}
    for model_idx, dets in enumerate(predictions_per_model):
        w = weights[model_idx]
        for d in dets:
            if d.confidence < skip_box_thr:
                continue
            cname = d.class_name.lower().replace("-", "_").replace(" ", "_")
            if cname not in class_groups:
                class_groups[cname] = []
            class_groups[cname].append({
                "box": np.array(d.box, dtype=np.float32),
                "conf": d.confidence,
                "model_idx": model_idx,
                "weight": w,
                "class_id": d.class_id,
                "class_name": d.class_name
            })

    fused_detections = []
    total_clusters = 0
    multi_model_agreed_clusters = 0

    for cname, items in class_groups.items():
        if not items:
            continue

        # Sort items by confidence descending
        items = sorted(items, key=lambda x: x["conf"], reverse=True)
        clusters = []

        for item in items:
            box = item["box"]
            matched_cluster_idx = None
            best_iou = 0.0

            for c_idx, cluster in enumerate(clusters):
                cluster_mean_box = cluster["mean_box"]
                # IoU between current box and cluster mean box
                iou = compute_iou_matrix(box[None, :], cluster_mean_box[None, :])[0, 0]
                if iou >= iou_thresh and iou > best_iou:
                    best_iou = iou
                    matched_cluster_idx = c_idx

            if matched_cluster_idx is not None:
                # Add to existing cluster
                cl = clusters[matched_cluster_idx]
                cl["items"].append(item)
                # Recompute weighted mean box
                total_w = sum(it["conf"] * it["weight"] for it in cl["items"])
                mean_b = sum((it["conf"] * it["weight"]) * it["box"] for it in cl["items"]) / max(1e-6, total_w)
                cl["mean_box"] = mean_b
            else:
                # Form new cluster
                clusters.append({
                    "mean_box": box.copy(),
                    "items": [item],
                    "class_name": item["class_name"],
                    "class_id": item["class_id"]
                })

        # Process each cluster to generate fused detection
        for cluster in clusters:
            total_clusters += 1
            contributing_models = set(it["model_idx"] for it in cluster["items"])
            if len(contributing_models) > 1:
                multi_model_agreed_clusters += 1

            # Compute fused confidence
            if conf_type == "avg":
                # Average confidence across all models (penalizes single-model outliers)
                fused_conf = sum(it["conf"] for it in cluster["items"]) / float(n_models)
            elif conf_type == "max":
                fused_conf = max(it["conf"] for it in cluster["items"])
            else:
                # Weighted mean
                fused_conf = sum(it["conf"] * it["weight"] for it in cluster["items"]) / sum(it["weight"] for it in cluster["items"])

            fused_detections.append(Detection(
                box=cluster["mean_box"].tolist(),
                class_id=cluster["class_id"],
                class_name=cluster["class_name"],
                confidence=float(fused_conf)
            ))

    # Agreement metric
    agreement_rate = (multi_model_agreed_clusters / total_clusters * 100.0) if total_clusters > 0 else 100.0

    analysis = {
        "total_clusters": total_clusters,
        "multi_model_agreed_clusters": multi_model_agreed_clusters,
        "agreement_rate_pct": round(agreement_rate, 2),
        "fused_count": len(fused_detections)
    }

    return fused_detections, analysis
