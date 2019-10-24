import unittest

from rest_framework.test import APITestCase, APIClient

from src.jobs.models import Job
from src.utils.tests_utils import create_test_split, create_test_log


class TestViews(APITestCase):
    def test_get_jobs(self):
        jobs = Job.objects.all()
        client = APIClient()
        response = client.get('/jobs/')
        self.assertEqual(len(jobs), len(response.data))
        client.post('/jobs/')
        jobs = Job.objects.all()
        response = client.get('/jobs/')
        self.assertEqual(len(jobs), len(response.data))

    def test_get_jobs_filtered_type(self):
        jobs = Job.objects.all()
        client = APIClient()
        response = client.get('/jobs/', {'type': 'prediction'})
        self.assertIsNotNone(response.data)

    def test_get_jobs_filtered_status(self):
        jobs = Job.objects.all()
        client = APIClient()
        response = client.get('/jobs/', {'status': 'created'})
        self.assertIsNotNone(response.data)

    def test_delete(self):
        client = APIClient()
        response = client.post('/jobs/', {}, format='json')
        db_id = Job.objects.all()[0].id
        self.assertEqual(db_id, dict(response.data)['id'])

        response = client.post('/jobs/', {'id': 1}, format='json')
        self.assertNotEqual(db_id, dict(response.data)['id'])

    @unittest.skip('needs refactoring')
    def test_create_multiple(self):
        create_test_split(original_log=create_test_log())

        client = APIClient()
        response = client.post('/jobs/multiple', {
            'type': 'classification',
            'split_id': 1,
            'config': {
                'clusterings': ['noCluster'],
                'encodings': ['simpleIndex'],
                'encoding': {
                    'padding': False,
                    'prefix_length': 1,
                    'generation_type': 'only',
                    'add_remaining_time': False,
                    'add_elapsed_time': False,
                    'add_executed_events': False,
                    'add_resources_used': False,
                    'add_new_traces': False,
                    'features': [],
                },
                'create_models': False,
                'methods': ['randomForest', 'decisionTree', 'rnn'],
                'kmeans': {},
                'incremental_train': {
                    'base_model': None,
                },
                'hyperparameter_optimizer': {
                    'algorithm_type': 'tpe',
                    'max_evaluations': 10,
                    'performance_metric': 'rmse',
                    'type': 'none',
                },
                'labelling': {
                    'type': 'next_activity',
                    'attribute_name': '',
                    'threshold_type': 'threshold_mean',
                    'threshold': 0,
                }
            }}, format='json')

        self.assertEqual(3, len(response.data))
