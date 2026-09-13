from typing import Dict, Any

class ModelManager:
    """Manages AI model loading status and operational mode (PROTOTYPE vs TRAINED AI MODE)."""
    def __init__(self):
        self.mode = "PROTOTYPE MODE (Rule-based Pose landmarker)"
        self.pose_status = "LOADED (MediaPipe Tasks)"
        self.object_status = "LOADED (Geometric Detector)"
        self.har_status = "PROTOTYPE / RULE-BASED"
        self.hmr_3d_status = "NOT ENABLED (Modular Stub)"

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "pose_model": self.pose_status,
            "object_model": self.object_status,
            "har_model": self.har_status,
            "hmr_3d_model": self.hmr_3d_status,
        }
