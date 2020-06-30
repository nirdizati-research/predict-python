import contextlib
from os import remove
from shutil import copyfile

from django.test import TestCase
from rest_framework import status
from rest_framework.test import APITestCase, APIClient

from src.logs.models import Log
from src.split.models import Split
from src.logs.log_service import get_log
from src.utils.tests_utils import general_example_filepath, general_example_test_filepath_xes, \
    general_example_test_filepath_csv


class LogModelTest(TestCase):
    def setUp(self):
        Log.objects.create(name="general_example.xes", path=general_example_filepath)

    def test_can_find_log_file(self):
        log = Log.objects.get(name="general_example.xes", path=general_example_filepath)

        log_file = get_log(log)

        self.assertEqual(6, len(log_file))


class SplitModelTest(TestCase):
    def setUp(self):
        log = Log.objects.create(name="general_example.xes", path=general_example_filepath)
        Split.objects.create(original_log=log)

    def test_can_find_split_original_file(self):
        log = Log.objects.get(name="general_example.xes", path=general_example_filepath)

        split = Split.objects.get(original_log=log)
        log_file = get_log(split.original_log)

        self.assertEqual(6, len(log_file))

    def test_to_dict(self):
        log = Log.objects.get(name="general_example.xes", path=general_example_filepath)

        split = Split.objects.get(original_log=log).to_dict()
        self.assertEqual('single', split['type'])
        self.assertEqual(general_example_filepath, split['original_log_path'])


class FileUploadTests(APITestCase):
    def tearDown(self):
        Log.objects.all().delete()
        # I hate that Python can't just delete
        with contextlib.suppress(FileNotFoundError):
            remove('cache/log_cache/test_upload.xes')  # TODO: fixme a file is not uploaded with only its name,
        with contextlib.suppress(FileNotFoundError):  # it is its name + the time.time() in order to avoid
            remove('cache/log_cache/file1.xes')  # shadowing and it is also sha265 encoded
        with contextlib.suppress(FileNotFoundError):
            remove('cache/log_cache/file2.xes')

    @staticmethod
    def _create_test_file_xes(path):
        copyfile(general_example_test_filepath_xes, path)
        f = open(path, 'rb')
        return f

    @staticmethod
    def _create_test_file_csv(path):
        copyfile(general_example_test_filepath_csv, path)
        f = open(path, 'rb')
        return f

    def test_upload_file_xes(self):
        f = self._create_test_file_xes('/tmp/test_upload.xes')

        client = APIClient()
        response = client.post('/logs/', {'single': f}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['name'][:11], 'test_upload')
        self.assertEqual(response.data['name'][-4:], '.xes')
        self.assertIsNotNone(response.data['properties']['events'])
        self.assertIsNotNone(response.data['properties']['resources'])
        self.assertIsNotNone(response.data['properties']['traceAttributes'])
        self.assertIsNotNone(response.data['properties']['maxEventsInLog'])
        self.assertIsNotNone(response.data['properties']['newTraces'])

    def test_upload_file_csv(self):
        f = self._create_test_file_csv('/tmp/test_upload.csv')

        client = APIClient()
        response = client.post('/logs/', {'single': f}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(response.data['name'][:11], 'test_upload')
        self.assertEqual(response.data['name'][-4:], '.csv')
        self.assertIsNotNone(response.data['properties']['events'])
        self.assertIsNotNone(response.data['properties']['resources'])
        self.assertIsNotNone(response.data['properties']['traceAttributes'])
        self.assertIsNotNone(response.data['properties']['maxEventsInLog'])
        self.assertIsNotNone(response.data['properties']['newTraces'])

    def test_upload_multiple_files_xes(self):
        f1 = self._create_test_file_xes('/tmp/file1.xes')
        f2 = self._create_test_file_xes('/tmp/file2.xes')

        client = APIClient()
        response = client.post('/splits/multiple', {'testSet': f1, 'trainingSet': f2}, format='multipart')
        self.assertEqual(response.data['type'], 'double')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIsNotNone(response.data['test_log'])
        self.assertIsNotNone(response.data['training_log'])
        self.assertIsNone(response.data['original_log'])

    def test_upload_multiple_files_csv(self):
        f1 = self._create_test_file_csv('/tmp/file1.csv')
        f2 = self._create_test_file_csv('/tmp/file2.csv')

        client = APIClient()
        response = client.post('/splits/multiple', {'testSet': f1, 'trainingSet': f2}, format='multipart')
        self.assertEqual(response.data['type'], 'double')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIsNotNone(response.data['test_log'])
        self.assertIsNotNone(response.data['training_log'])
        self.assertIsNone(response.data['original_log'])
