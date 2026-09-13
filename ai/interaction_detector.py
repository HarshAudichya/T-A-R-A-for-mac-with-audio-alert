import numpy as np
from typing import List, Dict, Any

class InteractionDetector:
    """Analyzes spatial hand-object interaction states (NOT_INTERACTING, NEAR_OBJECT, TOUCHING_OBJECT, HOLDING_OBJECT, MOVING_OBJECT)."""
    
    NOT_INTERACTING = "NOT_INTERACTING"
    NEAR_OBJECT = "NEAR_OBJECT"
    TOUCHING_OBJECT = "TOUCHING_OBJECT"
    HOLDING_OBJECT = "HOLDING_OBJECT"
    MOVING_OBJECT = "MOVING_OBJECT"

    def __init__(self):
        self.touch_threshold = 0.14
        self.near_threshold = 0.28

    def calculate_distance(self, hand_point: Any, obj_bbox: List[int], img_w: int, img_h: int) -> float:
        x, y, bw, bh = obj_bbox
        obj_center_x = (x + bw / 2.0) / img_w
        obj_center_y = (y + bh / 2.0) / img_h
        return float(np.hypot(hand_point.x - obj_center_x, hand_point.y - obj_center_y))

    def detect_interaction(self, landmarks: List[Any], detections: List[Dict[str, Any]], img_w: int, img_h: int) -> Dict[str, Any]:
        result = {
            "state": self.NOT_INTERACTING,
            "target_object": None,
            "min_distance": 999.0,
            "interaction_event": "None",
        }
        if not landmarks or len(landmarks) < 17 or not detections:
            return result

        left_wrist = landmarks[15]
        right_wrist = landmarks[16]

        for det in detections:
            bbox = det["bbox"]
            cls_name = det["class"]

            d_left = self.calculate_distance(left_wrist, bbox, img_w, img_h)
            d_right = self.calculate_distance(right_wrist, bbox, img_w, img_h)
            min_d = min(d_left, d_right)

            if min_d < result["min_distance"]:
                result["min_distance"] = min_d
                result["target_object"] = cls_name

                if min_d < 0.09:
                    result["state"] = self.HOLDING_OBJECT
                    result["interaction_event"] = f"Holding {cls_name}"
                elif min_d < self.touch_threshold:
                    result["state"] = self.TOUCHING_OBJECT
                    result["interaction_event"] = f"Touching {cls_name}"
                elif min_d < self.near_threshold:
                    result["state"] = self.NEAR_OBJECT
                    result["interaction_event"] = f"Near {cls_name}"
                else:
                    result["state"] = self.NOT_INTERACTING
                    result["interaction_event"] = f"Far from {cls_name}"

        return result
