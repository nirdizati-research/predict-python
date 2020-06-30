from django.test import TestCase

from src.logs.log_service import get_log
from src.utils.log_metrics import events_by_date, resources_by_date, event_executions, new_trace_start, \
    trace_attributes, events_in_trace, max_events_in_log, trace_ids_in_log, traces_in_log
from src.utils.tests_utils import general_example_filepath, financial_log_filepath, general_example_filename, \
    create_test_log, financial_log_filename


class LogTest(TestCase):

    def setUp(self):
        self.log = get_log(create_test_log(log_name=general_example_filename,
                                           log_path=general_example_filepath))

    def test_events_by_date(self):
        result = events_by_date(self.log)
        self.assertEqual(18, len(result.keys()))
        self.assertEqual(4, result['2011-01-08'])

    def test_resources_by_date(self):
        result = resources_by_date(self.log)
        self.assertEqual(18, len(result.keys()))
        self.assertEqual(4, result['2010-12-30'])
        self.assertEqual(3, result['2011-01-08'])
        self.assertEqual(1, result['2011-01-20'])

    def test_event_executions(self):
        result = event_executions(self.log)
        self.assertEqual(8, len(result.keys()))
        self.assertEqual(9, result['decide'])
        self.assertEqual(3, result['reject request'])

    def test_new_trace_start(self):
        result = new_trace_start(self.log)
        self.assertEqual(2, len(result.keys()))
        self.assertEqual(3, result['2010-12-30'])
        self.assertEqual(3, result['2011-01-06'])

    def test_trace_attributes(self):
        self.log = get_log(create_test_log(log_name=financial_log_filename,
                                           log_path=financial_log_filepath))
        result = trace_attributes(self.log)
        self.assertEqual(2, len(result))
        self.assertDictEqual({'name': 'AMOUNT_REQ', 'type': 'number', 'example': '20000'},
                             result[0])
        self.assertDictEqual({'name': 'REG_DATE', 'type': 'string', 'example': '2011-10-01 00:38:44.546000+02:00'},
                             result[1])

    def test_events_in_trace(self):
        result = events_in_trace(self.log)
        self.assertEqual(6, len(result.keys()))
        self.assertEqual(9, result['3'])

    def test_max_events_in_log(self):
        result = max_events_in_log(self.log)
        self.assertEqual(13, result)

    def test_trace_ids_in_log(self):
        result = trace_ids_in_log(self.log)
        self.assertEqual(6, len(result))
        self.assertEqual('4', result[5])

    def test_traces_in_log(self):
        result = traces_in_log(self.log)
        self.assertEqual(6, len(result))
        self.assertEqual({'concept:name': '3', 'creator': 'Fluxicon Nitro'}, result[0]['attributes'])
        self.assertEqual('Pete',result[0]['events'][0]['Resource'])
        self.assertEqual(9, len(result[0]['events']))
