import os
import glob
import json
import stat
import shutil
import threading
from datetime import datetime
import cv2
import numpy as np

def _remove_readonly(func, path, exc_info):
    """Windows permission error handler forcing write permissions on locked/read-only files."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        print(f"[DatasetGenerator] Remove warning on {path}: {e}")

class DatasetGenerator:
    """Group-isolated dataset generator saving high-res image frames and JSON metadata annotations without blocking video streaming."""
    def __init__(self, base_dir: str = "data/dataset", config_path: str = "config/activities.json"):
        self.base_dir = base_dir
        self.config_path = config_path
        self.activities_config = self._load_config()
        
        self.active_group_name = "Multi-Object Experiment"
        self.active_dir = os.path.join(self.base_dir, self.normalize_group_name(self.active_group_name))
        os.makedirs(self.active_dir, exist_ok=True)
        self.labels = self._get_labels_for_group(self.active_group_name)

    def _load_config(self) -> dict:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[DatasetGenerator] Error loading config: {e}")
        return {}

    def normalize_group_name(self, group_name: str) -> str:
        return group_name.lower().replace(" ", "_")

    def normalize_label(self, label: str) -> str:
        return label.lower().replace(" ", "_")

    def set_active_group(self, group_name: str) -> None:
        self.active_group_name = group_name
        self.active_dir = os.path.join(self.base_dir, self.normalize_group_name(group_name))
        os.makedirs(self.active_dir, exist_ok=True)
        self.labels = self._get_labels_for_group(group_name)

    def _get_labels_for_group(self, group_name: str) -> list:
        groups = self.activities_config.get("activity_groups", self.activities_config)
        if group_name in groups:
            group_obj = groups[group_name]
            steps = group_obj.get("sequence", group_obj.get("steps", []))
            labels = [s["activity"] for s in steps]
            if labels:
                return labels
        return ["stand", "sit", "raise_both_hands", "raise_right_hand", "raise_left_hand", "hands_together"]

    def save_sample(self, label: str, frame: np.ndarray, metadata: dict = None) -> bool:
        """Saves high-res image frame (.jpg) and metadata annotation (.json) asynchronously in background."""
        if frame is None:
            return False

        norm_label = self.normalize_label(label)
        target_dir = os.path.join(self.active_dir, norm_label)
        os.makedirs(target_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        img_filename = f"{norm_label}_{timestamp}.jpg"
        json_filename = f"{norm_label}_{timestamp}.json"

        img_path = os.path.join(target_dir, img_filename)
        json_path = os.path.join(target_dir, json_filename)
        frame_copy = frame.copy()

        def _write_async():
            try:
                cv2.imwrite(img_path, frame_copy)
                meta_content = {
                    "timestamp": datetime.now().isoformat(),
                    "activity_group": self.active_group_name,
                    "activity_label": label,
                    "normalized_label": norm_label,
                    "image_file": img_filename,
                    "metadata": metadata or {}
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta_content, f, indent=2)
                print(f"[DatasetGenerator] Saved dataset sample: {img_path}")
            except Exception as e:
                print(f"[DatasetGenerator] Error writing sample: {e}")

        threading.Thread(target=_write_async, daemon=True).start()
        return True

    def get_statistics(self, group_name: str = None) -> dict:
        target_group = group_name or self.active_group_name
        group_dir = os.path.join(self.base_dir, self.normalize_group_name(target_group))
        labels = self._get_labels_for_group(target_group)
        
        stats = {}
        for lbl in labels:
            norm_lbl = self.normalize_label(lbl)
            lbl_dir = os.path.join(group_dir, norm_lbl)
            legacy_dir = os.path.join(self.base_dir, norm_lbl)
            
            cnt = 0
            if os.path.exists(lbl_dir):
                cnt += len(glob.glob(os.path.join(lbl_dir, "*.jpg")))
            if os.path.exists(legacy_dir):
                cnt += len(glob.glob(os.path.join(legacy_dir, "*.jpg")))
            stats[lbl] = cnt
        return stats

    def clear_dataset(self, group_only: bool = False) -> bool:
        """Clears all dataset files and subfolders cleanly on Windows without WinError 5 Access Denied."""
        try:
            target_path = self.active_dir if group_only else self.base_dir
            if os.path.exists(target_path):
                for root, dirs, files in os.walk(target_path, topdown=False):
                    for f in files:
                        p = os.path.join(root, f)
                        try:
                            os.chmod(p, stat.S_IWRITE)
                            os.remove(p)
                        except Exception:
                            pass
                    for d in dirs:
                        p = os.path.join(root, d)
                        try:
                            os.chmod(p, stat.S_IWRITE)
                            shutil.rmtree(p, onerror=_remove_readonly)
                        except Exception:
                            pass
                try:
                    shutil.rmtree(target_path, onerror=_remove_readonly)
                except Exception:
                    pass

            os.makedirs(self.base_dir, exist_ok=True)
            os.makedirs(self.active_dir, exist_ok=True)
            print(f"[DatasetGenerator] Dataset storage at {target_path} successfully reset to 0 samples!")
            return True
        except Exception as e:
            print(f"[DatasetGenerator] Error clearing dataset: {e}")
            return False
