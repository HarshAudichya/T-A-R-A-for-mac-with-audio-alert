import json
import os
import csv
from datetime import datetime
from typing import List, Dict, Any

class LogManager:
    """Stores timestamped experiment events, saves JSON/CSV logs to data/logs/."""
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.events: List[Dict[str, Any]] = []

    def log_event(self, step_id: int, expected: str, detected: str, confidence: float, status: str) -> Dict[str, Any]:
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step_id": step_id,
            "expected_activity": expected,
            "detected_activity": detected,
            "confidence": round(float(confidence), 2),
            "status": status,
        }
        self.events.append(entry)
        self.auto_save()
        return entry

    def auto_save(self) -> None:
        if not self.events:
            return
        ts = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(self.log_dir, f"experiment_log_{ts}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.events, f, indent=2)
        except Exception:
            pass

    def get_events(self) -> List[Dict[str, Any]]:
        return self.events

    def get_json_data(self) -> str:
        return json.dumps(self.events, indent=2)

    def clear(self) -> None:
        self.events = []
