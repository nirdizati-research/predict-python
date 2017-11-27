from django.test import TestCase
from rest_framework import status
from rest_framework.test import APITestCase, APIClient

from jobs.models import Job
from logs.models import Log, Split


class JobModelTest(TestCase):
    def setUp(self):
        self.config = {'key': 123}
        log = Log.objects.create(name="general_example.xes", path="log_cache/general_example.xes")
        split = Split.objects.create(original_log=log)
        Job.objects.create(config=self.config, split=split)

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
