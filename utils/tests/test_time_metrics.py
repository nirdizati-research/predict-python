from unittest import TestCase

from utils import log_metrics
from utils.time_metrics import *
from logs.file_service import get_log


class TimeMetrics(TestCase):
    def setUp(self):
        self.log = get_log("log_cache/general_example.xes")

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

    def test_count_on_event_day(self):
        event_dict = log_metrics.events_by_date(self.log)
        count = count_on_event_day(self.log[0], event_dict, 0)
        self.assertEqual(7, count)

    def test_count_on_event_day_no_such_date(self):
        count = count_on_event_day(self.log[0], dict(), 0)
        self.assertEqual(0, count)

    def test_count_on_event_day_event_out_of_range(self):
        event_dict = log_metrics.events_by_date(self.log)
        count = count_on_event_day(self.log[0], event_dict, 100)
        self.assertEqual(0, count)
