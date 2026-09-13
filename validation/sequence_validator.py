from typing import Dict, Any, List, Optional

class SequenceValidator:
    """Validates detected activities and required object presence against expected sequence steps."""
    
    CORRECT = "CORRECT"
    WAITING = "WAITING"
    WRONG_ACTIVITY = "WRONG_ACTIVITY"
    OBJECT_MISSING = "OBJECT_MISSING"
    OUT_OF_SEQUENCE = "OUT_OF_SEQUENCE"
    SKIPPED = "SKIPPED"
    COMPLETED = "COMPLETED"

    def validate(
        self,
        expected_activity: Optional[str],
        expected_object: Optional[str],
        detected_activity: str,
        detected_objects: List[Dict[str, Any]],
        confidence: float
    ) -> Dict[str, Any]:
        """Compares expected step activity & required object with detected activity & detected object list."""
        if not expected_activity or expected_activity == "None (Completed)":
            return {
                "status": self.COMPLETED,
                "message": "Experiment Sequence Completed Successfully!",
                "is_valid": True,
            }

        if detected_activity == "Unknown" or confidence < 0.70:
            return {
                "status": self.WAITING,
                "message": f"Waiting for step: '{expected_activity}'...",
                "is_valid": False,
            }

        # Validate Required Object Presence if step requires an object
        if expected_object:
            detected_class_names = [d["class"] for d in detected_objects]
            if expected_object not in detected_class_names:
                return {
                    "status": self.OBJECT_MISSING,
                    "message": f"⚠ Required Object Missing: '{expected_object}' not detected in scene!",
                    "is_valid": False,
                }

        # Validate Activity Match
        if detected_activity == expected_activity:
            return {
                "status": self.CORRECT,
                "message": f"✓ CORRECT STEP: {detected_activity}",
                "is_valid": True,
            }
        else:
            return {
                "status": self.WRONG_ACTIVITY,
                "message": f"⚠ Incorrect Activity: Detected '{detected_activity}' (Expected '{expected_activity}')",
                "is_valid": False,
            }
