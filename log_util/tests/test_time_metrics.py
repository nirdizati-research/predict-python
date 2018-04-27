from unittest import TestCase

from log_util.time_metrics import duration, elapsed_time_id, remaining_time_id
from logs.file_service import get_logs


class TimeMetrics(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example.xes")[0]

    def test_calculate_remaining_time(self):
        trace = self.log[0]
        seconds = remaining_time_id(trace, 4)
        self.assertEqual(772020.0, seconds)

        seconds = remaining_time_id(trace, 8)
        self.assertEqual(0.0, seconds)

    def test_calculate_elapsed_time(self):
        trace = self.log[0]
        seconds = elapsed_time_id(trace, 4)
        self.assertEqual(596760.0, seconds)

        seconds = elapsed_time_id(trace, 0)
        self.assertEqual(0.0, seconds)

    def test_calculate_duration(self):
        seconds = duration(self.log[0])
        self.assertEqual(1368780.0, seconds)

        seconds = duration(self.log[1])
        self.assertEqual(779580.0, seconds)


