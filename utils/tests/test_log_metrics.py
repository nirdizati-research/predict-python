from django.test import TestCase

from utils.file_service import get_log
from utils.log_metrics import *
from utils.tests_utils import general_example_filepath, financial_log_filepath


class LogTest(TestCase):
    def test_events_by_date(self):
        logs = get_log(general_example_filepath)
        result = events_by_date(logs)
        self.assertEqual(18, len(result.keys()))
        self.assertEqual(4, result['2011-01-08'])

    def test_resources_by_date(self):
        logs = get_log(general_example_filepath)
        result = resources_by_date(logs)
        self.assertEqual(18, len(result.keys()))
        self.assertEqual(4, result['2010-12-30'])
        self.assertEqual(3, result['2011-01-08'])
        self.assertEqual(1, result['2011-01-20'])

    def test_event_executions(self):
        logs = get_log(general_example_filepath)
        result = event_executions(logs)
        self.assertEqual(8, len(result.keys()))
        self.assertEqual(9, result['decide'])
        self.assertEqual(3, result['reject request'])

    def test_new_trace_start(self):
        logs = get_log(general_example_filepath)
        result = new_trace_start(logs)
        self.assertEqual(2, len(result.keys()))
        self.assertEqual(3, result['2010-12-30'])
        self.assertEqual(3, result['2011-01-06'])

    def test_trace_attributes(self):
        logs = get_log(financial_log_filepath)
        result = trace_attributes(logs)
        self.assertEqual(2, len(result))
        self.assertDictEqual({'name': 'AMOUNT_REQ', 'type': 'number', 'example': '20000'},
                             result[0])
        self.assertDictEqual({'name': 'REG_DATE', 'type': 'string', 'example': '2011-10-01 00:38:44.546000+02:00'},
                             result[1])

    def test_events_in_trace(self):
        logs = get_log(general_example_filepath)
        result = events_in_trace(logs)
        self.assertEqual(6, len(result.keys()))
        self.assertEqual(9, result['3'])

    def test_max_events_in_log(self):
        logs = get_log(general_example_filepath)
        result = max_events_in_log(logs)
        self.assertEqual(13, result)
