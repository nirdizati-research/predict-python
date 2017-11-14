from unittest import TestCase

from encoders.log_util import unique_events, elapsed_time_id, remaining_time_id
from logs.file_service import get_logs


class TestSimpleGeneralExample(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example.xes")[0]

    def test_unique_events(self):
        events = unique_events(self.log)
        self.assertEqual(8, len(events))

    def test_calculate_remaining_time(self):
        trace = self.log[0]
        seconds = remaining_time_id(trace, 4)
        self.assertEqual(772020.0, seconds)

        # last event
        trace2 = self.log[0]
        seconds = remaining_time_id(trace2, 8)
        self.assertEqual(0.0, seconds)

    def test_calculate_elapsed_time(self):
        trace = self.log[0]
        seconds = elapsed_time_id(trace, 4)
        self.assertEqual(596760.0, seconds)

        # first event
        trace2 = self.log[0]
        seconds = elapsed_time_id(trace2, 0)
        self.assertEqual(0.0, seconds)

    def test_mxml_gz(self):
        log = get_logs("log_cache/nonlocal.mxml.gz")[0]
        events = unique_events(log)
        self.assertEqual(7, len(events))
