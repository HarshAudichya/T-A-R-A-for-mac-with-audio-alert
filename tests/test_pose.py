import unittest
import numpy as np
from ai.pose_estimator import PoseEstimator
from ai.object_detector import ObjectDetector
from ai.interaction_detector import InteractionDetector
from ai.activity_recognizer import ActivityRecognizer

class TestAIPipeline(unittest.TestCase):
    def test_object_detector_and_interaction(self):
        detector = ObjectDetector()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add red box in center
        dummy_frame[200:300, 200:300] = [0, 0, 255]
        detections = detector.detect(dummy_frame)
        self.assertTrue(isinstance(detections, list))

        interactor = InteractionDetector()
        res = interactor.detect_interaction([], detections, 640, 480)
        self.assertIn("interaction_event", res)

    def test_activity_recognizer(self):
        rec = ActivityRecognizer()
        activity, conf = rec.predict([])
        self.assertEqual(activity, "Unknown")
        self.assertEqual(conf, 0.0)

if __name__ == "__main__":
    unittest.main()
