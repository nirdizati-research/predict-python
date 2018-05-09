from unittest import TestCase

from log_util.event_attributes import unique_events, unique_events2, get_event_attributes, get_global_event_attributes
from logs.file_service import get_logs


class EventAttributes(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example.xes")[0]

    def test_unique_events(self):
        events = unique_events(self.log)
        self.assertEqual(8, len(events))

    def test_mxml_gz(self):
        log = get_logs("log_cache/nonlocal.mxml.gz")[0]
        events = unique_events(log)
        self.assertEqual(7, len(events))

    def test_multiple_unique_events(self):
        test_log = get_logs("log_cache/general_example_test.xes")[0]
        training_log = get_logs("log_cache/general_example_training.xes")[0]
        events = unique_events2(training_log, test_log)
        self.assertEqual(8, len(events))

    def test_event_attributes(self):
        log = get_logs("log_cache/general_example_test.xes")[0]
        attributes = get_event_attributes(log)
        self.assertListEqual(attributes, ['Activity', 'Costs', 'Resource', 'org:resource'])

    def test_global_event_attributes(self):
        log = get_logs("log_cache/general_example_test.xes")[0]
        attributes = get_global_event_attributes(log)
        self.assertListEqual(attributes, ['Activity', 'Costs', 'Resource', 'org:resource'])
