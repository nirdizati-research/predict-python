from os import remove

from django.test import SimpleTestCase, TestCase
from rest_framework import status
from rest_framework.test import APITestCase, APIClient

from logs.models import Log, Split
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


class SplitModelTest(TestCase):
    def setUp(self):
        log = Log.objects.create(name="general_example.xes", path="log_cache/general_example.xes")
        Split.objects.create(original_log=log)

    def test_can_find_split_original_file(self):
        """Split file can be found by id"""
        split = Split.objects.get(id=1)
        log_file = split.original_log.get_file()

        self.assertEqual(1, len(log_file))
        self.assertEqual(6, len(log_file[0]))
        self.assertEqual({}, split.config)


class FileUploadTests(APITestCase):
    def tearDown(self):
        Log.objects.all().delete()
        remove('log_cache/test_upload')

    def _create_test_file(self, path):
        f = open(path, 'w')
        f.write('test123\n')
        f.close()
        f = open(path, 'rb')
        return {'file': f}

    def test_upload_file(self):
        data = self._create_test_file('/tmp/test_upload')

        client = APIClient()
        response = client.post('/logs/', data, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['name'], 'test_upload')
