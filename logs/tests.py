from django.test import SimpleTestCase, TestCase

from logs.models import Log
from .file_service import get_logs
from .log_service import events_by_date, resources_by_date, event_executions


class LogTest(SimpleTestCase):
    def test_events_by_date(self):
        logs = get_logs("log_cache/general_example.xes")
        result = events_by_date(logs)
        self.assertEqual(18, len(result.keys()))
        self.assertEqual(4, result['2011-01-08'])

    def test_resources_by_date(self):
        logs = get_logs("log_cache/general_example.xes")
        result = resources_by_date(logs)
        self.assertEqual(18, len(result.keys()))
        self.assertEqual(4, result['2010-12-30'])
        self.assertEqual(3, result['2011-01-08'])
        self.assertEqual(1, result['2011-01-20'])

    def test_event_executions(self):
        logs = get_logs("log_cache/general_example.xes")
        result = event_executions(logs)
        self.assertEqual(8, len(result.keys()))
        self.assertEqual(9, result['decide'])
        self.assertEqual(3, result['reject request'])


class LogModelTest(TestCase):
    def setUp(self):
        Log.objects.create(name="general_example.xes", path="log_cache/general_example.xes")

    def test_can_find_log_file(self):
        """Log file can be found by id"""
        log = Log.objects.get(id=1)
        log_file = log.get_file()

        self.assertEqual(1, len(log_file))
        self.assertEqual(6, len(log_file[0]))
