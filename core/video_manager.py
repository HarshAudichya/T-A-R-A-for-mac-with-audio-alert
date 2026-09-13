import cv2
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

class VideoManager:
    """Manages MP4 video recording with dynamic resolution auto-detection and robust multi-codec fallback."""
    def __init__(self, output_dir: str = "data/videos"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.is_recording = False
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_filepath = ""
        self.current_filename = ""
        self.start_time = 0.0
        self.frames_written = 0
        self.fps = 20.0
        self.preset_w = 0
        self.preset_h = 0

    def start_recording(self, width: int = 0, height: int = 0, fps: float = 20.0) -> bool:
        """Flags recording to start; lazy-initializes VideoWriter on the first written frame."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_filename = f"experiment_{timestamp}.mp4"
        self.current_filepath = os.path.join(self.output_dir, self.current_filename)
        self.is_recording = True
        self.start_time = time.time()
        self.frames_written = 0
        self.fps = fps
        self.preset_w = width
        self.preset_h = height
        self.writer = None
        print(f"[VideoManager] Recording flagged ON. File target: {self.current_filepath}")
        return True

    def write_frame(self, frame: Any) -> bool:
        """Writes a BGR frame to disk, initializing VideoWriter lazily to match exact frame dimensions."""
        if not self.is_recording or frame is None:
            return False

        h, w, _ = frame.shape
        target_w = self.preset_w if self.preset_w > 0 else w
        target_h = self.preset_h if self.preset_h > 0 else h

        if self.writer is None:
            # Try mp4v, XVID, MJPG codecs
            codecs = ["mp4v", "XVID", "MJPG"]
            success = False
            for c in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*c)
                    self.writer = cv2.VideoWriter(self.current_filepath, fourcc, self.fps, (target_w, target_h))
                    if self.writer.isOpened():
                        print(f"[VideoManager] Successfully initialized VideoWriter with codec '{c}' at {target_w}x{target_h}")
                        success = True
                        break
                except Exception as e:
                    print(f"[VideoManager] Codec '{c}' failed: {e}")

            if not success or not self.writer.isOpened():
                print("[VideoManager] Failed to open any VideoWriter codec.")
                self.is_recording = False
                return False

        try:
            if (target_w, target_h) != (w, h):
                frame = cv2.resize(frame, (target_w, target_h))
            self.writer.write(frame)
            self.frames_written += 1
            return True
        except Exception as e:
            print(f"[VideoManager] Error writing frame: {e}")
            return False

    def stop_recording(self) -> Dict[str, Any]:
        """Stops recording session and releases video file writer handle."""
        self.is_recording = False
        duration = time.time() - self.start_time if self.start_time > 0 else 0.0
        if self.writer:
            self.writer.release()
            self.writer = None
            print(f"[VideoManager] Stopped recording. Saved {self.frames_written} frames ({duration:.1f}s) to {self.current_filepath}")

        return {
            "filename": self.current_filename,
            "filepath": self.current_filepath,
            "frames": self.frames_written,
            "duration_sec": round(duration, 1),
        }
