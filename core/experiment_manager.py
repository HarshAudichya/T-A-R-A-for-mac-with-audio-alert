from typing import List, Dict, Any, Optional
from core.activity_manager import ActivityManager

class ExperimentManager:
    """Manages ordered experiment steps, step sequence advancement, and active activity group sequence configuration."""
    def __init__(self, config_path: str = "config/activities.json"):
        self.activity_manager = ActivityManager(config_path)
        self.current_idx = 0
        self.steps: List[Dict[str, Any]] = []
        self.load_current_group_sequence()

    def load_current_group_sequence(self) -> None:
        self.steps = self.activity_manager.get_current_sequence()
        self.current_idx = 0

    def set_activity_group(self, group_name: str) -> bool:
        ok = self.activity_manager.set_activity_group(group_name)
        if ok:
            self.load_current_group_sequence()
        return ok

    def reset(self) -> None:
        self.current_idx = 0

    def is_completed(self) -> bool:
        return self.current_idx >= len(self.steps)

    def get_current_step(self) -> Optional[Dict[str, Any]]:
        if self.current_idx < len(self.steps):
            return self.steps[self.current_idx]
        return None

    def get_expected_activity(self) -> Optional[str]:
        step = self.get_current_step()
        return step["activity"] if step else None

    def get_expected_required_object(self) -> Optional[str]:
        step = self.get_current_step()
        return step.get("required_object") if step else None

    def get_next_step(self) -> Optional[Dict[str, Any]]:
        if self.current_idx + 1 < len(self.steps):
            return self.steps[self.current_idx + 1]
        return None

    def advance_step(self) -> bool:
        if self.current_idx < len(self.steps):
            self.current_idx += 1
            return True
        return False

    def get_progress_info(self) -> Dict[str, Any]:
        return {
            "group_name": self.activity_manager.get_current_group(),
            "current_step": self.current_idx + 1 if not self.is_completed() else len(self.steps),
            "total_steps": len(self.steps),
            "completed": self.is_completed(),
            "expected_activity": self.get_expected_activity() or "None (Completed)",
            "required_objects": self.activity_manager.get_required_objects(),
        }
