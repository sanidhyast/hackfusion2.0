"""
PPE Compliance and Spatial Association Engine
Associates detected safety equipment with individual workers using anatomical sub-zoning
and calculates per-worker compliance and violation summaries.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2

# Class definitions from Construction-PPE dataset
CLASSES = {
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

@dataclass
class WorkerStatus:
    worker_id: int
    box: List[float]  # [x1, y1, x2, y2]
    confidence: float
    is_compliant: bool
    missing_ppe: List[str]
    detected_ppe: Dict[str, Optional[Detection]] = field(default_factory=dict)
    negative_flags: List[str] = field(default_factory=list)
    verdict: str = "VIOLATION"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "box": [round(c, 2) for c in self.box],
            "confidence": round(self.confidence, 3),
            "verdict": self.verdict,
            "is_compliant": self.is_compliant,
            "missing_ppe": self.missing_ppe,
            "detected_ppe": {
                k: {"box": [round(c, 2) for c in v.box], "conf": round(v.confidence, 3)}
                for k, v in self.detected_ppe.items() if v is not None
            },
            "negative_flags": self.negative_flags
        }

def compute_box_intersection(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    return inter_w * inter_h

def compute_ioa(item_box: List[float], zone_box: List[float]) -> float:
    """Intersection over Item Area: How much of the item lies inside the target zone."""
    item_w = max(0.0, item_box[2] - item_box[0])
    item_h = max(0.0, item_box[3] - item_box[1])
    item_area = item_w * item_h
    if item_area <= 0:
        return 0.0
    inter_area = compute_box_intersection(item_box, zone_box)
    return inter_area / item_area

def extract_anatomical_zones(person_box: List[float]) -> Dict[str, List[float]]:
    """
    Splits person bounding box into anatomical target zones:
    - Head Zone: Top 0% to 35% for upright workers, expanding up to 50% for leaning/reaching/overhead postures
    - Torso Zone: 18% to 80% of height
    - Full Body: 0% to 100%
    """
    x1, y1, x2, y2 = person_box
    h = max(1.0, y2 - y1)
    w = max(1.0, x2 - x1)
    # For overhead/leaning/reaching workers (aspect ratio w/h > 0.75), head extends deeper into the box
    head_ratio = 0.50 if (w / h) > 0.75 else 0.35
    return {
        "head": [x1, y1, x2, y1 + head_ratio * h],
        "torso": [x1, y1 + 0.18 * h, x2, y1 + 0.80 * h],
        "full": [x1, y1, x2, y2]
    }

class PPEAssociationEngine:
    def __init__(
        self,
        required_ppe: Optional[List[str]] = None,
        head_ioa_threshold: float = 0.25,
        torso_ioa_threshold: float = 0.35,
        min_person_conf: float = 0.35,
        min_ppe_conf: float = 0.15
    ):
        # Default mandatory equipment per problem statement: helmet and safety vest
        self.required_ppe = required_ppe or ["helmet", "vest"]
        self.head_ioa_threshold = head_ioa_threshold
        self.torso_ioa_threshold = torso_ioa_threshold
        self.min_person_conf = min_person_conf
        self.min_ppe_conf = min_ppe_conf

    def associate_and_evaluate(
        self,
        detections: List[Detection]
    ) -> Tuple[List[WorkerStatus], List[Detection], Dict[str, Any]]:
        """
        Associates items to workers and computes compliance.
        Returns:
            - List of WorkerStatus
            - List of Unassigned Detections (e.g. helmet on table, in hand, loose vest)
            - Summary dictionary
        """
        # Normalize and separate workers from equipment and negative labels
        def norm_name(c: str) -> str:
            cn = c.lower().replace("-", "_").replace(" ", "_")
            if cn in {"person", "worker", "human"}:
                return "person"
            if cn in {"helmet", "hardhat", "hard_hat"}:
                return "helmet"
            if cn in {"vest", "safety_vest", "high_vis_vest"}:
                return "vest"
            if cn in {"no_helmet", "no_hardhat", "no_hard_hat"}:
                return "no_helmet"
            if cn in {"no_vest", "no_safety_vest"}:
                return "no_vest"
            return cn

        workers = [
            d for d in detections 
            if norm_name(d.class_name) == "person" and d.confidence >= self.min_person_conf
        ]
        
        helmets = [d for d in detections if norm_name(d.class_name) == "helmet" and d.confidence >= self.min_ppe_conf]
        vests = [d for d in detections if norm_name(d.class_name) == "vest" and d.confidence >= self.min_ppe_conf]
        no_helmets = [d for d in detections if norm_name(d.class_name) == "no_helmet" and d.confidence >= self.min_ppe_conf]
        
        unassigned_items = []
        worker_statuses = []

        # Sort workers left-to-right for consistent numbering
        workers = sorted(workers, key=lambda w: w.box[0])

        for idx, worker in enumerate(workers, 1):
            zones = extract_anatomical_zones(worker.box)
            detected_ppe = {}
            negative_flags = []

            # 1. Check Helmet association (must be on head zone)
            best_helmet = None
            best_helmet_ioa = 0.0
            for h in helmets:
                ioa = compute_ioa(h.box, zones["head"])
                if ioa > best_helmet_ioa and ioa >= self.head_ioa_threshold:
                    best_helmet = h
                    best_helmet_ioa = ioa
            
            if best_helmet:
                detected_ppe["helmet"] = best_helmet

            # 2. Check Vest association (must be on torso zone)
            best_vest = None
            best_vest_ioa = 0.0
            for v in vests:
                ioa = compute_ioa(v.box, zones["torso"])
                if ioa > best_vest_ioa and ioa >= self.torso_ioa_threshold:
                    best_vest = v
                    best_vest_ioa = ioa

            if best_vest:
                detected_ppe["vest"] = best_vest

            # 3. Check for explicit negative labels (e.g. no_helmet) in head zone
            # A negative label (bare head / no_helmet) only indicates violation if:
            # - No helmet was detected on the worker, OR
            # - The no_helmet box directly overlaps the detected helmet box AND has significantly higher confidence
            has_conflicting_no_helmet = False
            for nh in no_helmets:
                ioa = compute_ioa(nh.box, zones["head"])
                if ioa >= self.head_ioa_threshold:
                    if best_helmet is None:
                        has_conflicting_no_helmet = True
                        negative_flags.append("no_helmet")
                    else:
                        inter = compute_box_intersection(nh.box, best_helmet.box)
                        if inter > 0 and nh.confidence > (best_helmet.confidence + 0.15):
                            has_conflicting_no_helmet = True
                            negative_flags.append("no_helmet")

            # 4. Evaluate compliance rules
            missing = []
            for req in self.required_ppe:
                if req not in detected_ppe or detected_ppe[req] is None:
                    missing.append(f"missing_{req}")
                elif req == "helmet" and has_conflicting_no_helmet:
                    missing.append("missing_helmet")

            # Remove duplicates in missing
            missing = list(set(missing))
            is_compliant = len(missing) == 0
            verdict = "COMPLIANT" if is_compliant else "VIOLATION"

            worker_status = WorkerStatus(
                worker_id=idx,
                box=worker.box,
                confidence=worker.confidence,
                is_compliant=is_compliant,
                missing_ppe=missing,
                detected_ppe=detected_ppe,
                negative_flags=negative_flags,
                verdict=verdict
            )
            worker_statuses.append(worker_status)

        # Identify unassigned items (e.g. helmets held in hand or sitting on tables)
        associated_helmet_ids = {id(ws.detected_ppe["helmet"]) for ws in worker_statuses if ws.detected_ppe.get("helmet")}
        associated_vest_ids = {id(ws.detected_ppe["vest"]) for ws in worker_statuses if ws.detected_ppe.get("vest")}

        for h in helmets:
            if id(h) not in associated_helmet_ids:
                unassigned_items.append(h)
        for v in vests:
            if id(v) not in associated_vest_ids:
                unassigned_items.append(v)

        # Check if unassigned helmet is carried by a worker (under arm, in hand, or on lap)
        for ws in worker_statuses:
            if ws.detected_ppe.get("helmet") is None:
                wx1, wy1, wx2, wy2 = ws.box
                wh = max(1.0, wy2 - wy1)
                ww = max(1.0, wx2 - wx1)
                # Spatial arm / hand / waist / lap carrying zone (extends slightly to catch cropped bodies & hand positions)
                carry_zone = [wx1 - 0.35 * ww, wy1 + 0.18 * wh, wx2 + 0.35 * ww, wy2 + 0.30 * wh]
                for h in unassigned_items:
                    ioa = compute_ioa(h.box, carry_zone)
                    inter = compute_box_intersection(h.box, carry_zone)
                    if norm_name(h.class_name) == "helmet" and (ioa > 0.15 or (h.area > 0 and inter / h.area > 0.15)):
                        if "helmet_under_arm" not in ws.negative_flags:
                            ws.negative_flags.append("helmet_under_arm")
                        if "missing_helmet" in ws.missing_ppe:
                            ws.missing_ppe.remove("missing_helmet")
                        if "helmet_under_arm" not in ws.missing_ppe:
                            ws.missing_ppe.append("helmet_under_arm")
                        ws.is_compliant = False
                        ws.verdict = "VIOLATION"
                        break

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
    """Draws rich visual overlays with worker boxes, status badges, and connector lines."""
    annotated = image.copy()
    h_img, w_img = annotated.shape[:2]

    # Draw unassigned items with amber warning color
    for item in unassigned_items:
        x1, y1, x2, y2 = [int(v) for v in item.box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2, lineType=cv2.LINE_AA)
        label = f"Unworn {item.class_name}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)

    # Draw workers
    for ws in worker_statuses:
        x1, y1, x2, y2 = [int(v) for v in ws.box]
        color = (0, 200, 0) if ws.is_compliant else (0, 0, 230)  # Green vs Red (BGR)

        # Person bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3, lineType=cv2.LINE_AA)

        # Status Badge at top
        verdict_str = f"Worker #{ws.worker_id}: {ws.verdict}"
        if not ws.is_compliant:
            missing_str = ", ".join([m.replace("missing_", "No ") for m in ws.missing_ppe])
            badge_text = f"{verdict_str} [{missing_str}]"
        else:
            badge_text = f"{verdict_str} [Helmet + Vest]"

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - th - 12)
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated, badge_text, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw associated equipment boxes and link lines
        for ppe_type, ppe_det in ws.detected_ppe.items():
            if ppe_det:
                px1, py1, px2, py2 = [int(v) for v in ppe_det.box]
                ppe_color = (0, 255, 128) if ppe_type == "helmet" else (255, 200, 0)
                cv2.rectangle(annotated, (px1, py1), (px2, py2), ppe_color, 2, lineType=cv2.LINE_AA)
                
                # Connector line from worker center to PPE center
                w_cx, w_cy = (x1 + x2) // 2, (y1 + y2) // 2
                p_cx, p_cy = (px1 + px2) // 2, (py1 + py2) // 2
                cv2.line(annotated, (w_cx, w_cy), (p_cx, p_cy), ppe_color, 1, cv2.LINE_AA)

    return annotated
