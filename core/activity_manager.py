import json
import os
from typing import List, Dict, Any, Optional

class ActivityManager:
    """Manages Activity Groups, active group switching, required objects, and sequence loading."""
    def __init__(self, config_path: str = "config/activities.json"):
        self.config_path = config_path
        self.groups_data: Dict[str, Any] = {}
        self.current_group_name = "Basic Human Activities"
        self.load_config()

    def load_config(self) -> None:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.groups_data = json.load(f)
            except Exception as e:
                print(f"[ActivityManager] Config load error: {e}")
                self._load_fallback()
        else:
            self._load_fallback()

    def _load_fallback(self) -> None:
        self.groups_data = {
            "Basic Human Activities": {
                "description": "Standard unassisted body pose activities",
                "required_objects": [],
                "sequence": [
                    { "step": 1, "activity": "Stand", "instruction": "Stand normally", "required_object": None },
                    { "step": 2, "activity": "Raise Both Hands", "instruction": "Raise both hands", "required_object": None },
                    { "step": 3, "activity": "Hands Together", "instruction": "Bring hands together", "required_object": None },
                    { "step": 4, "activity": "Raise Right Hand", "instruction": "Raise right hand", "required_object": None },
                    { "step": 5, "activity": "Sit", "instruction": "Sit down", "required_object": None },
                    { "step": 6, "activity": "Stand", "instruction": "Stand up", "required_object": None },
                    { "step": 7, "activity": "Raise Left Hand", "instruction": "Raise left hand", "required_object": None }
                ]
            }
        }

    def get_activity_groups(self) -> List[str]:
        return list(self.groups_data.keys())

    def set_activity_group(self, group_name: str) -> bool:
        if group_name in self.groups_data:
            self.current_group_name = group_name
            return True
        return False

    def get_current_group(self) -> str:
        return self.current_group_name

    def get_group_info(self, group_name: str = None) -> Dict[str, Any]:
        target = group_name or self.current_group_name
        return self.groups_data.get(target, {})

    def get_activities(self) -> List[str]:
        info = self.get_group_info()
        seq = info.get("sequence", [])
        return [item["activity"] for item in seq]

    def get_required_objects(self) -> List[str]:
        info = self.get_group_info()
        return info.get("required_objects", [])

    def get_current_sequence(self) -> List[Dict[str, Any]]:
        info = self.get_group_info()
        return info.get("sequence", [])
