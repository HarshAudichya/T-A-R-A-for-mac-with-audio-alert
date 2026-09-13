import unittest
from logging_system.log_manager import LogManager

class TestLogManager(unittest.TestCase):
    def test_log_event(self):
        lm = LogManager(log_dir="data/logs")
        entry = lm.log_event(1, "Stand", "Stand", 0.95, "COMPLETED")
        self.assertEqual(entry["step_id"], 1)
        self.assertEqual(len(lm.events), 1)

        json_data = lm.get_json_data()
        self.assertIn("Stand", json_data)

if __name__ == "__main__":
    unittest.main()
