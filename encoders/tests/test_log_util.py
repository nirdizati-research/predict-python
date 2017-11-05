from unittest import TestCase

from encoders.log_util import unique_events, calculate_remaining_time, calculate_elapsed_time
from logs.file_service import get_logs


class TestSimpleGeneralExample(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example.xes")[0]

    def test_unique_events(self):
        events = unique_events(self.log)
        self.assertEqual(8, len(events))

    def test_calculate_remaining_time(self):
        trace = self.log[0]
        seconds = calculate_remaining_time(trace, 4)
        self.assertEqual(772020.0, seconds)

        # last event
        trace2 = self.log[0]
        seconds = calculate_remaining_time(trace2, 8)
        self.assertEqual(0.0, seconds)

    def test_calculate_elapsed_time(self):
        trace = self.log[0]
        seconds = calculate_elapsed_time(trace, 4)
        self.assertEqual(596760.0, seconds)

        # first event
        trace2 = self.log[0]
        seconds = calculate_elapsed_time(trace2, 0)
        self.assertEqual(0.0, seconds)
