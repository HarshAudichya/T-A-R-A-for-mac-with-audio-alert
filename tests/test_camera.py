import unittest
import numpy as np
from core.camera_manager import CameraManager
from core.video_manager import VideoManager

class TestCameraAndVideo(unittest.TestCase):
    def test_camera_manager_metrics(self):
        cm = CameraManager()
        cm.update_frame_metrics()
        status = cm.get_status()
        self.assertTrue(status["connected"])
        self.assertGreaterEqual(status["frame_count"], 1)

    def test_video_manager_recording(self):
        vm = VideoManager(output_dir="data/videos")
        started = vm.start_recording(width=100, height=100, fps=10.0)
        self.assertTrue(started)
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        vm.write_frame(dummy_frame)
        self.assertEqual(vm.frames_written, 1)
        vm.stop_recording()
        self.assertFalse(vm.is_recording)

if __name__ == "__main__":
    unittest.main()
