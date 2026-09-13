import unittest
import numpy as np
import cv2
from core.activity_manager import ActivityManager
from core.experiment_manager import ExperimentManager
from ai.object_detector import ObjectDetector
from telemetry.bandwidth_telemetry import BandwidthTelemetryTracker
from reports.report_generator import MissionReportGenerator
from alerts.alert_manager import AlertManager

class TestISROFeatures(unittest.TestCase):
    def test_isro_red_yellow_box_experiment_loaded(self):
        am = ActivityManager("config/activities.json")
        groups = am.get_activity_groups()
        self.assertIn("ISRO Red & Yellow Box BAS Experiment", groups)
        self.assertIn("Multi-Object Experiment", groups)
        self.assertIn("Basic Human Activities", groups)

        em = ExperimentManager("config/activities.json")
        em.set_activity_group("ISRO Red & Yellow Box BAS Experiment")
        self.assertEqual(len(em.steps), 7)
        self.assertEqual(em.get_expected_activity(), "Approach Outer Box")
        self.assertEqual(em.get_expected_required_object(), "Box")

    def test_visual_color_classification_red_and_yellow(self):
        detector = ObjectDetector()
        
        # Create solid red image (BGR: 0, 0, 255)
        red_roi = np.zeros((100, 100, 3), dtype=np.uint8)
        red_roi[:, :] = (0, 0, 255)
        self.assertEqual(detector.classify_box_color(red_roi), "Red Box")

        # Create solid yellow image (BGR: 0, 255, 255)
        yellow_roi = np.zeros((100, 100, 3), dtype=np.uint8)
        yellow_roi[:, :] = (0, 255, 255)
        self.assertEqual(detector.classify_box_color(yellow_roi), "Yellow Box")

        # Create neutral gray/black image (BGR: 50, 50, 50)
        neutral_roi = np.zeros((100, 100, 3), dtype=np.uint8)
        neutral_roi[:, :] = (50, 50, 50)
        self.assertEqual(detector.classify_box_color(neutral_roi), "Box")

    def test_bandwidth_telemetry_tracker(self):
        tracker = BandwidthTelemetryTracker()
        res = tracker.compute_telemetry(width=640, height=480, fps=30.0)
        self.assertGreater(res["raw_video_rate_mb_s"], 10.0)
        self.assertGreater(res["data_saved_pct"], 99.0)
        self.assertIn("ACTIVE", res["edge_status"])

    def test_alert_manager_voice_event(self):
        am = AlertManager(cooldown_seconds=0.1)
        msg = am.trigger_correct_step_alert("Open Outer Box")
        ev = am.get_latest_voice_event()
        self.assertIsNotNone(ev)
        self.assertEqual(ev["type"], "voice_alert")
        self.assertIn("Open Outer Box", ev["message"])

    def test_mission_audit_report_generation(self):
        steps = [
            {"step": 1, "activity": "Open Outer Box", "required_object": "Box"},
            {"step": 2, "activity": "Extract Red Box", "required_object": "Red Box"}
        ]
        events = [
            {"step": 1, "expected_activity": "Open Outer Box", "detected_activity": "Open Outer Box", "confidence": 0.95, "status": "COMPLETED", "timestamp": "2026-09-03 09:00:00"}
        ]
        html = MissionReportGenerator.generate_html_report(
            experiment_name="ISRO Red & Yellow Box BAS Experiment",
            steps=steps,
            current_idx=1,
            events=events
        )
        self.assertIn("TARA", html)
        self.assertIn("ISRO Red & Yellow Box BAS Experiment", html)
        self.assertIn("COMPLETED", html)

if __name__ == "__main__":
    unittest.main()
