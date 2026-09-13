import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Tuple, Any, Optional

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

class PoseEstimator:
    """Python 3.14 compatible MediaPipe Tasks Pose Estimator with Shared Singleton Initialization."""
    _shared_landmarker = None

    def __init__(self, model_path: str = "models/pose_landmarker_lite.task"):
        self.model_path = model_path
        self._ensure_model_file()
        if PoseEstimator._shared_landmarker is None:
            self._init_landmarker()

    def _ensure_model_file(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if not os.path.exists(self.model_path):
            print(f"[PoseEstimator] Downloading pose landmarker model to {self.model_path}...")
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
                print("[PoseEstimator] Download complete!")
            except Exception as e:
                print(f"[PoseEstimator] Error downloading model: {e}")

    def _init_landmarker(self) -> None:
        try:
            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            PoseEstimator._shared_landmarker = vision.PoseLandmarker.create_from_options(options)
            print("[PoseEstimator] Shared MediaPipe landmarker created successfully!")
        except Exception as e:
            print(f"[PoseEstimator] Landmarker init error: {e}")

    def estimate(self, image: np.ndarray) -> Tuple[List[Any], np.ndarray]:
        """Runs pose estimation on BGR OpenCV image and returns (landmarks, annotated_image)."""
        annotated = image.copy()
        if PoseEstimator._shared_landmarker is None:
            return [], annotated

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = PoseEstimator._shared_landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return [], annotated

            landmarks = result.pose_landmarks[0]
            h, w, _ = image.shape

            # Draw pose skeleton connections
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                (24, 26), (26, 28)
            ]
            for start_idx, end_idx in connections:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    p1 = landmarks[start_idx]
                    p2 = landmarks[end_idx]
                    vis1 = getattr(p1, 'visibility', getattr(p1, 'presence', 1.0))
                    vis2 = getattr(p2, 'visibility', getattr(p2, 'presence', 1.0))
                    if vis1 > 0.4 and vis2 > 0.4:
                        x1, y1 = int(p1.x * w), int(p1.y * h)
                        x2, y2 = int(p2.x * w), int(p2.y * h)
                        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for lm in landmarks:
                vis = getattr(lm, 'visibility', getattr(lm, 'presence', 1.0))
                if vis > 0.4:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

            return landmarks, annotated
        except Exception as e:
            print(f"[PoseEstimator] Estimation error: {e}")
            return [], annotated
