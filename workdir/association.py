"""
YOLOv8-Strict 5-Class PPE Spatial Association & Compliance Engine
Fixed Classes:
  0: Person
  1: Helmet
  2: No Helmet
  3: Vest
  4: No Vest
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2

CLASSES = {
    0: "Person",
    1: "Helmet",
    2: "No Helmet",
    3: "Vest",
    4: "No Vest"
}

@dataclass
class Detection:
    box: List[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float

    @property
    def area(self) -> float:
        w = max(0.0, self.box[2] - self.box[0])
        h = max(0.0, self.box[3] - self.box[1])
        return w * h

    @property
    def center(self) -> Tuple[float, float]:
        cx = (self.box[0] + self.box[2]) / 2.0
        cy = (self.box[1] + self.box[3]) / 2.0
        return (cx, cy)


@dataclass
class WorkerStatus:
    worker_id: int
    box: List[float]  # [x1, y1, x2, y2]
    confidence: float
    is_compliant: bool
    verdict: str  # COMPLIANT, VIOLATION, UNCERTAIN
    missing_ppe: List[str]
    detected_ppe: Dict[str, Optional[Detection]] = field(default_factory=dict)
    negative_flags: List[str] = field(default_factory=list)
    uncertainty_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "box": [round(c, 2) for c in self.box],
            "confidence": round(self.confidence, 3),
            "verdict": self.verdict,
            "is_compliant": self.is_compliant,
            "missing_ppe": self.missing_ppe,
            "uncertainty_reason": self.uncertainty_reason,
            "detected_ppe": {
                k: {"box": [round(c, 2) for c in v.box], "conf": round(v.confidence, 3)}
                for k, v in self.detected_ppe.items() if v is not None
            },
            "negative_flags": self.negative_flags
        }


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0.0


def compute_ioa(item_box: List[float], zone_box: List[float]) -> float:
    item_w = max(0.0, item_box[2] - item_box[0])
    item_h = max(0.0, item_box[3] - item_box[1])
    item_area = item_w * item_h
    if item_area <= 0:
        return 0.0
    xA = max(item_box[0], zone_box[0])
    yA = max(item_box[1], zone_box[1])
    xB = min(item_box[2], zone_box[2])
    yB = min(item_box[3], zone_box[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    return inter / item_area


def suppress_duplicates(dets: List[Detection], iou_thresh: float = 0.45) -> List[Detection]:
    if not dets:
        return []
    sorted_dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept = []
    for d in sorted_dets:
        discard = False
        for k in kept:
            if compute_iou(d.box, k.box) >= iou_thresh:
                discard = True
                break
        if not discard:
            kept.append(d)
    return kept


def resolve_contradictory_detections(
    positive_dets: List[Detection],
    negative_dets: List[Detection],
    iou_thresh: float = 0.30
) -> Tuple[List[Detection], List[Detection]]:
    kept_pos = []
    kept_neg = []
    resolved_neg_ids = set()

    for pos in positive_dets:
        suppress_pos = False
        for neg in negative_dets:
            if compute_iou(pos.box, neg.box) >= iou_thresh:
                if neg.confidence > (pos.confidence + 0.05):
                    suppress_pos = True
                    resolved_neg_ids.add(id(neg))
                    break
                else:
                    resolved_neg_ids.add(id(neg))
        if not suppress_pos:
            kept_pos.append(pos)

    for neg in negative_dets:
        if id(neg) not in resolved_neg_ids:
            kept_neg.append(neg)

    return kept_pos, kept_neg


def extract_anatomical_zones(person_box: List[float]) -> Dict[str, List[float]]:
    x1, y1, x2, y2 = person_box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    aspect = h / w

    if aspect > 1.7:
        head_bottom = y1 + 0.22 * h
        torso_top = y1 + 0.18 * h
        torso_bottom = y1 + 0.70 * h
    else:
        head_bottom = y1 + 0.28 * h
        torso_top = y1 + 0.22 * h
        torso_bottom = y1 + 0.85 * h

    return {
        "head": [x1, y1, x2, head_bottom],
        "torso": [x1, torso_top, x2, torso_bottom],
        "full": [x1, y1, x2, y2]
    }


class PPEAssociationEngine:
    def __init__(
        self,
        required_ppe: Optional[List[str]] = None,
        min_person_conf: float = 0.30,
        min_ppe_conf: float = 0.20,
        min_person_height_px: float = 50.0
    ):
        self.required_ppe = required_ppe or ["helmet", "vest"]
        self.min_person_conf = min_person_conf
        self.min_ppe_conf = min_ppe_conf
        self.min_person_height_px = min_person_height_px

    def associate_and_evaluate(
        self,
        detections: List[Detection]
    ) -> Tuple[List[WorkerStatus], List[Detection], Dict[str, Any]]:
        # 1. Filter workers
        workers = [
            d for d in detections
            if d.class_id == 0 and d.confidence >= self.min_person_conf
        ]
        workers = sorted(workers, key=lambda w: w.box[0])

        # 2. Extract equipment detections
        raw_helmets = [d for d in detections if d.class_id == 1 and d.confidence >= self.min_ppe_conf]
        raw_no_helmets = [d for d in detections if d.class_id == 2 and d.confidence >= self.min_ppe_conf]
        raw_vests = [d for d in detections if d.class_id == 3 and d.confidence >= self.min_ppe_conf]
        raw_no_vests = [d for d in detections if d.class_id == 4 and d.confidence >= self.min_ppe_conf]

        helmets = suppress_duplicates(raw_helmets, iou_thresh=0.40)
        no_helmets = suppress_duplicates(raw_no_helmets, iou_thresh=0.40)
        vests = suppress_duplicates(raw_vests, iou_thresh=0.40)
        no_vests = suppress_duplicates(raw_no_vests, iou_thresh=0.40)

        # Resolve contradictory overlap
        helmets, no_helmets = resolve_contradictory_detections(helmets, no_helmets, iou_thresh=0.30)
        vests, no_vests = resolve_contradictory_detections(vests, no_vests, iou_thresh=0.30)

        assigned_helmet_ids = set()
        assigned_vest_ids = set()
        unassigned_items = []
        worker_statuses = []

        for idx, worker in enumerate(workers, 1):
            wx1, wy1, wx2, wy2 = worker.box
            ww = max(1.0, wx2 - wx1)
            wh = max(1.0, wy2 - wy1)
            w_cx, w_cy = worker.center
            aspect = wh / ww

            if wh < self.min_person_height_px:
                ws = WorkerStatus(
                    worker_id=idx,
                    box=worker.box,
                    confidence=worker.confidence,
                    is_compliant=False,
                    verdict="UNCERTAIN",
                    missing_ppe=[],
                    uncertainty_reason="Worker resolution too low / distant"
                )
                worker_statuses.append(ws)
                continue

            zones = extract_anatomical_zones(worker.box)
            detected_ppe = {}
            negative_flags = []
            missing_ppe = []

            # Head / Helmet evaluation (must be on head zone)
            max_hy1 = wy1 + (0.16 * wh if aspect > 1.7 else 0.20 * wh)
            max_h_cy = wy1 + (0.25 * wh if aspect > 1.7 else 0.30 * wh)

            worn_helmet = None
            candidate_helmets = []
            for h in helmets:
                if id(h) in assigned_helmet_ids:
                    continue
                hx1, hy1, hx2, hy2 = h.box
                h_cx, h_cy = h.center
                if abs(h_cx - w_cx) / ww > 0.35:
                    continue
                if hy1 <= max_hy1 and h_cy <= max_h_cy:
                    ioa = compute_ioa(h.box, zones["head"])
                    if ioa >= 0.20:
                        candidate_helmets.append((ioa * h.confidence, h))

            if candidate_helmets:
                candidate_helmets.sort(key=lambda x: x[0], reverse=True)
                worn_helmet = candidate_helmets[0][1]
                detected_ppe["helmet"] = worn_helmet
                assigned_helmet_ids.add(id(worn_helmet))

            # Bare head detection (Class 2: No Helmet)
            has_bare_head = False
            for nh in no_helmets:
                nh_x1, nh_y1, nh_x2, nh_y2 = nh.box
                nh_cx, nh_cy = nh.center
                if abs(nh_cx - w_cx) / ww > 0.35:
                    continue
                if nh_y1 <= max_hy1 and nh_cy <= max_h_cy:
                    ioa = compute_ioa(nh.box, zones["head"])
                    if ioa >= 0.20:
                        has_bare_head = True
                        negative_flags.append("no_helmet")
                        break

            # Torso / Vest evaluation (must be on torso zone)
            worn_vest = None
            candidate_vests = []
            for v in vests:
                if id(v) in assigned_vest_ids:
                    continue
                vx1, vy1, vx2, vy2 = v.box
                v_cx, v_cy = v.center
                if abs(v_cx - w_cx) / ww > 0.38:
                    continue
                ioa = compute_ioa(v.box, zones["torso"])
                if ioa >= 0.25:
                    candidate_vests.append((ioa * v.confidence, v))

            if candidate_vests:
                candidate_vests.sort(key=lambda x: x[0], reverse=True)
                worn_vest = candidate_vests[0][1]
                detected_ppe["vest"] = worn_vest
                assigned_vest_ids.add(id(worn_vest))

            # Bare torso detection (Class 4: No Vest)
            has_bare_torso = False
            for nv in no_vests:
                nv_x1, nv_y1, nv_x2, nv_y2 = nv.box
                nv_cx, nv_cy = nv.center
                if abs(nv_cx - w_cx) / ww > 0.38:
                    continue
                ioa = compute_ioa(nv.box, zones["torso"])
                if ioa >= 0.25:
                    has_bare_torso = True
                    negative_flags.append("no_vest")
                    break

            # Determine compliance strictly against fixed classes:
            if "helmet" in self.required_ppe:
                if worn_helmet is None or has_bare_head:
                    missing_ppe.append("missing_helmet")

            if "vest" in self.required_ppe:
                if worn_vest is None or has_bare_torso:
                    missing_ppe.append("missing_vest")

            is_compliant = (len(missing_ppe) == 0)
            verdict = "COMPLIANT" if is_compliant else "VIOLATION"

            ws = WorkerStatus(
                worker_id=idx,
                box=worker.box,
                confidence=worker.confidence,
                is_compliant=is_compliant,
                verdict=verdict,
                missing_ppe=list(set(missing_ppe)),
                detected_ppe=detected_ppe,
                negative_flags=negative_flags
            )
            worker_statuses.append(ws)

        # Unassigned equipment
        for h in helmets:
            if id(h) not in assigned_helmet_ids:
                unassigned_items.append(h)
        for v in vests:
            if id(v) not in assigned_vest_ids:
                unassigned_items.append(v)
        for nh in no_helmets:
            unassigned_items.append(nh)
        for nv in no_vests:
            unassigned_items.append(nv)

        total_workers = len(worker_statuses)
        compliant_count = sum(1 for w in worker_statuses if w.is_compliant)
        violation_count = total_workers - compliant_count

        summary = {
            "total_workers": total_workers,
            "compliant_workers": compliant_count,
            "violation_workers": violation_count,
            "compliance_rate": round((compliant_count / total_workers * 100), 1) if total_workers > 0 else 100.0,
            "unassigned_items_count": len(unassigned_items),
            "unassigned_items": [
                {
                    "box": [round(c, 2) for c in u.box],
                    "class_name": u.class_name,
                    "confidence": round(u.confidence, 3)
                }
                for u in unassigned_items
            ],
            "workers": [w.to_dict() for w in worker_statuses]
        }

        return worker_statuses, unassigned_items, summary


def draw_compliance_annotations(
    image: np.ndarray,
    worker_statuses: List[WorkerStatus],
    unassigned_items: List[Detection]
) -> np.ndarray:
    annotated = image.copy()
    h_img, w_img = annotated.shape[:2]

    for ws in worker_statuses:
        x1, y1, x2, y2 = [int(v) for v in ws.box]

        if ws.verdict == "COMPLIANT":
            color = (0, 200, 0)
        elif ws.verdict == "UNCERTAIN":
            color = (0, 165, 255)
        else:
            color = (0, 0, 230)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3, lineType=cv2.LINE_AA)

        if ws.verdict == "COMPLIANT":
            badge_text = f"Worker #{ws.worker_id} - Compliant ({int(ws.confidence * 100)}%)"
        elif ws.verdict == "UNCERTAIN":
            badge_text = f"Worker #{ws.worker_id} - Uncertain: Low Res"
        else:
            violations_text = []
            if "missing_helmet" in ws.missing_ppe:
                violations_text.append("No Helmet")
            if "missing_vest" in ws.missing_ppe:
                violations_text.append("No Vest")
            viol_str = ", ".join(violations_text) if violations_text else "Violation"
            badge_text = f"Worker #{ws.worker_id} - {viol_str} ({int(ws.confidence * 100)}%)"

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - th - 12)
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, badge_text, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        for ppe_type, ppe_det in ws.detected_ppe.items():
            if ppe_det:
                px1, py1, px2, py2 = [int(v) for v in ppe_det.box]
                ppe_color = (0, 255, 128) if ppe_type == "helmet" else (255, 200, 0)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), ppe_color, 2, lineType=cv2.LINE_AA)
                cv2.putText(annotated, ppe_type.capitalize(), (px1, max(15, py1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, ppe_color, 1, cv2.LINE_AA)

    for item in unassigned_items:
        x1, y1, x2, y2 = [int(v) for v in item.box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(annotated, f"Unworn {item.class_name}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

    return annotated
