import unittest
from core.activity_manager import ActivityManager
from core.experiment_manager import ExperimentManager
from validation.sequence_validator import SequenceValidator

class TestSequenceValidation(unittest.TestCase):
    def test_activity_manager_groups(self):
        am = ActivityManager("config/activities.json")
        groups = am.get_activity_groups()
        self.assertIn("Basic Human Activities", groups)
        self.assertIn("Multi-Object Experiment", groups)
        self.assertIn("Phone Activities", groups)

        am.set_activity_group("Multi-Object Experiment")
        objs = am.get_required_objects()
        self.assertIn("Phone", objs)
        self.assertIn("Bottle", objs)
        self.assertIn("Chair", objs)

    def test_experiment_manager_sequence(self):
        em = ExperimentManager("config/activities.json")
        em.set_activity_group("Multi-Object Experiment")
        self.assertEqual(len(em.steps), 9)
        self.assertEqual(em.get_expected_activity(), "Pick Up Phone")
        self.assertEqual(em.get_expected_required_object(), "Phone")

    def test_sequence_validator_with_objects(self):
        sv = SequenceValidator()
        
        # Valid activity + object present
        res_ok = sv.validate("Pick Up Phone", "Phone", "Pick Up Phone", [{"class": "Phone"}], 0.95)
        self.assertEqual(res_ok["status"], SequenceValidator.CORRECT)

        # Missing object
        res_missing = sv.validate("Pick Up Phone", "Phone", "Pick Up Phone", [{"class": "Bottle"}], 0.95)
        self.assertEqual(res_missing["status"], SequenceValidator.OBJECT_MISSING)

        # Wrong activity
        res_wrong = sv.validate("Pick Up Phone", "Phone", "Sit On Chair", [{"class": "Phone"}], 0.95)
        self.assertEqual(res_wrong["status"], SequenceValidator.WRONG_ACTIVITY)

if __name__ == "__main__":
    unittest.main()
