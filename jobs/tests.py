from django.test import TestCase
from rest_framework import status
from rest_framework.test import APITestCase, APIClient

from core.constants import CLASSIFICATION, REGRESSION
from jobs.models import Job
from jobs.tasks import prediction_task
from logs.models import Log, Split


class JobModelTest(TestCase):
    def setUp(self):
        self.config = {'key': 123,
                       'method': 'randomForest',
                       'encoding': 'simpleIndex',
                       'clustering': 'none',
                       "rule": "remaining_time",
                       "prefix_length": 1,
                       "threshold": "default"
                       }
        log = Log.objects.create(name="general_example.xes", path="log_cache/general_example.xes")
        split = Split.objects.create(original_log=log)
        Job.objects.create(config=self.config, split=split, type=CLASSIFICATION)
        Job.objects.create(config=self.config, split=split, type='asdsd')
        del self.config['method']
        Job.objects.create(config=self.config, split=split, type=REGRESSION)

    def test_default(self):
        job = Job.objects.get(id=1)

        self.assertEqual(self.config, job.config)
        self.assertEqual('created', job.status)
        self.assertIsNotNone(job.created_date)
        self.assertIsNotNone(job.modified_date)
        self.assertEqual({}, job.result)

    def test_modified(self):
        job = Job.objects.get(id=1)
        job.status = 'completed'

        self.assertNotEquals(job.created_date, job.modified_date)

    def test_to_dict(self):
        job = Job.objects.get(id=1).to_dict()

        self.assertEquals(CLASSIFICATION, job['type'])
        self.assertDictEqual({'type': 'single',
                              'original_log_path': "log_cache/general_example.xes",
                              'config': {}},
                             job['split'])
        self.assertEquals(123, job['key'])

    def test_prediction_task(self):
        prediction_task(1)

        job = Job.objects.get(id=1)

        self.assertEqual('completed', job.status)
        self.assertNotEqual({}, job.result)

    def test_prediction_task_error(self):
        self.assertRaises(ValueError, prediction_task, 2)
        job = Job.objects.get(id=2)

        self.assertEqual('error', job.status)
        self.assertEqual({}, job.result)
        self.assertEqual("ValueError('Type not supported', 'asdsd')", job.error)

    def test_missing_attributes(self):
        self.assertRaises(KeyError, prediction_task, 3)
        job = Job.objects.get(id=3)

        self.assertEqual('error', job.status)
        self.assertEqual({}, job.result)
        self.assertEqual("KeyError('method',)", job.error)


class CreateJobsTests(APITestCase):
    def setUp(self):
        log = Log.objects.create(name="general_example.xes", path="log_cache/general_example.xes")
        Split.objects.create(original_log=log)

    def job_obj(self):
        config = dict()
        config['encodings'] = ['simpleIndex']
        config['clusterings'] = ['none']
        config['methods'] = ['kmeans']
        config['random'] = 123
        obj = dict()
        obj['type'] = 'classification'
        obj['config'] = config
        obj['split_id'] = 1
        return obj

    def test_class_job_creation(self):
        client = APIClient()
        response = client.post('/jobs/multiple', self.job_obj(), format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['type'], 'classification')
        self.assertEqual(response.data[0]['config']['encoding'], 'simpleIndex')
        self.assertEqual(response.data[0]['config']['clustering'], 'none')
        self.assertEqual(response.data[0]['config']['method'], 'kmeans')
        self.assertEqual(response.data[0]['config']['random'], 123)
        self.assertEqual(response.data[0]['status'], 'created')

    def job_obj2(self):
        config = dict()
        config['encodings'] = ['simpleIndex', 'boolean', 'complex']
        config['clusterings'] = ['none']
        config['methods'] = ['linear', 'lasso']
        config['random'] = 123
        config['prefix_length'] = 1
        obj = dict()
        obj['type'] = 'regression'
        obj['config'] = config
        obj['split_id'] = 1
        return obj

    def test_reg_job_creation(self):
        client = APIClient()
        response = client.post('/jobs/multiple', self.job_obj2(), format='json')

        self.assertEqual(status.HTTP_201_CREATED, response.status_code)
        self.assertEqual(6, len(response.data), )
        self.assertEqual('regression', response.data[0]['type'])
        self.assertEqual('simpleIndex', response.data[0]['config']['encoding'])
        self.assertEqual('none', response.data[0]['config']['clustering'])
        self.assertEqual('linear', response.data[0]['config']['method'])
        self.assertEqual(123, response.data[0]['config']['random'])
        self.assertEqual('created', response.data[0]['status'])
        self.assertEqual(1, response.data[0]['split']['id'])
