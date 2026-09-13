import time
from typing import Dict, Any

class BandwidthTelemetryTracker:
    """Calculates and monitors Spacecraft Edge AI Bandwidth Optimization vs streaming raw video to Earth."""

    def __init__(self):
        self.last_update_time = time.time()
        self.edge_status = "ACTIVE (EDGE AI PROCESSING)"

    def compute_telemetry(self, width: int = 640, height: int = 480, fps: float = 30.0, json_payload_bytes: int = 1200) -> Dict[str, Any]:
        """Calculates raw video bitrate vs edge telemetry data rate and bandwidth savings percentage."""
        w = max(width, 320)
        h = max(height, 240)
        f = max(fps, 1.0)

        # Raw uncompressed video stream rate: Width * Height * 3 (BGR) * FPS
        raw_bytes_per_sec = float(w * h * 3 * f)
        raw_mb_s = raw_bytes_per_sec / (1024.0 * 1024.0)

        # Processed structured AI output data rate (JSON events, coordinates, metadata)
        # Emitted at ~1 structured event/telemetry packet per second or ~1.2 KB/s
        ai_bytes_per_sec = float(max(json_payload_bytes, 800))
        ai_kb_s = ai_bytes_per_sec / 1024.0

        # Percentage of bandwidth saved by running AI locally at the edge
        data_saved_pct = (1.0 - (ai_bytes_per_sec / raw_bytes_per_sec)) * 100.0

        return {
            "raw_video_rate_mb_s": round(raw_mb_s, 1),
            "raw_video_rate_str": f"{raw_mb_s:.1f} MB/s",
            "ai_telemetry_rate_kb_s": round(ai_kb_s, 2),
            "ai_telemetry_rate_str": f"{ai_kb_s:.1f} KB/s",
            "data_saved_pct": round(data_saved_pct, 2),
            "data_saved_str": f"{data_saved_pct:.2f}%",
            "edge_status": self.edge_status,
            "bandwidth_label": "ESTIMATED SPACECRAFT EDGE TELEMETRY"
        }
