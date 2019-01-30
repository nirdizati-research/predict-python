from unittest import TestCase

from log_util.event_attributes import unique_events, unique_events2, get_event_attributes, \
    get_additional_columns
from logs.file_service import get_log


class EventAttributes(TestCase):
    def setUp(self):
        self.log = get_log("log_cache/general_example.xes")

    def test_unique_events(self):
        events = unique_events(self.log)
        self.assertEqual(8, len(events))

    # def test_mxml_gz(self):
    #     log = get_log("log_cache/nonlocal.mxml.gz")
    #     events = unique_events(log)
    #     self.assertEqual(7, len(events))

    def test_multiple_unique_events(self):
        test_log = get_log("log_cache/general_example_test.xes")
        training_log = get_log("log_cache/general_example_training.xes")
        events = unique_events2(training_log, test_log)
        self.assertEqual(8, len(events))

    def test_event_attributes(self):
        log = get_log("log_cache/general_example_test.xes")
        attributes = get_event_attributes(log)
        self.assertListEqual(attributes, ['Activity', 'Costs', 'Resource', 'org:resource'])

    def test_global_event_attributes(self):
        log = get_log("log_cache/general_example_test.xes")
        attributes = get_additional_columns(log)
        self.assertListEqual(attributes['event_attributes'], ['Activity', 'Costs', 'Resource', 'org:resource'])
