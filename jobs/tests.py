from django.test import TestCase
from django_rq.queues import get_queue
from rest_framework import status
from rest_framework.test import APITestCase, APIClient

from core.constants import CLASSIFICATION, REGRESSION
from core.tests.test_prepare import add_default_config
from jobs.job_creator import create_config, _classification_random_forest
from jobs.models import Job
from jobs.tasks import prediction_task
from logs.models import Log, Split


class JobModelTest(TestCase):
    def setUp(self):
        self.config = {'key': 123,
                       'method': 'randomForest',
                       'encoding': 'simpleIndex',
                       'clustering': 'noCluster',
                       "rule": "remaining_time",
                       "prefix_length": 1,
                       "padding": 'no_padding',
                       "threshold": "default",
                       "create_models": False,
                       }
        log = Log.objects.create(name="general_example.xes", path="log_cache/general_example.xes")
        split = Split.objects.create(original_log=log)
        Job.objects.create(config=add_default_config(self.config, type=CLASSIFICATION), split=split,
                           type=CLASSIFICATION)
        Job.objects.create(config=self.config, split=split, type='asdsd')
        Job.objects.create(config={}, split=split, type=REGRESSION)

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
                              'config': {},
                              'id': 1},
                             job['split'])
        self.assertEquals(123, job['key'])

    def test_prediction_task(self):
        prediction_task(1)

        job = Job.objects.get(id=1)

        self.assertEqual('completed', job.status)
        self.assertNotEqual({}, job.result)

    def test_create_models_config_missing(self):
        job = Job.objects.get(id=1)
        del job.config["create_models"]
        job.save()
        prediction_task(1)

        job = Job.objects.get(id=1)
        print(job.config)

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


class Hyperopt(TestCase):
    def setUp(self):
        self.config = {
            'method': 'randomForest',
            'encoding': 'simpleIndex',
            'clustering': 'noCluster',
            "rule": "remaining_time",
            "prefix_length": 3,
            "padding": 'no_padding',
            "threshold": "default",
            "create_models": False,
            "hyperopt": {"use_hyperopt": True, "max_evals": 2, "performance_metric": "acc"}
        }
        log = Log.objects.create(name="general_example.xes", path="log_cache/general_example.xes")
        split = Split.objects.create(original_log=log)
        Job.objects.create(config=add_default_config(self.config, type=CLASSIFICATION), split=split,
                           type=CLASSIFICATION)

    def test_hyperopt(self):
        prediction_task(1)
        job = Job.objects.get(id=1)
        self.assertFalse(_classification_random_forest() == job.config['classification.randomForest'])


class CreateJobsTests(APITestCase):
    def setUp(self):
        log = Log.objects.create(name="general_example.xes", path="log_cache/general_example.xes")
        Split.objects.create(original_log=log)

    def tearDown(self):
        get_queue().empty()

    def job_obj(self):
        config = dict()
        config['encodings'] = ['simpleIndex']
        config['clusterings'] = ['noCluster']
        config['methods'] = ['knn']
        config['random'] = 123
        config['prefix'] = {'prefix_length': 3, 'type': 'only', 'padding': 'zero_padding'}
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
        self.assertEqual(response.data[0]['config']['clustering'], 'noCluster')
        self.assertEqual(response.data[0]['config']['method'], 'knn')
        self.assertEqual(response.data[0]['config']['random'], 123)
        self.assertEqual(response.data[0]['config']['prefix_length'], 3)
        self.assertEqual(response.data[0]['config']['padding'], 'zero_padding')
        self.assertEqual(response.data[0]['status'], 'created')

    def job_obj2(self):
        config = dict()
        config['encodings'] = ['simpleIndex', 'boolean', 'complex']
        config['clusterings'] = ['noCluster']
        config['methods'] = ['linear', 'lasso']
        config['random'] = 123
        config['prefix'] = {'prefix_length': 3, 'type': 'up_to', 'padding': 'no_padding'}
        obj = dict()
        obj['type'] = 'regression'
        obj['config'] = config
        obj['split_id'] = 1
        return obj

    def test_reg_job_creation(self):
        client = APIClient()
        response = client.post('/jobs/multiple', self.job_obj2(), format='json')

        self.assertEqual(status.HTTP_201_CREATED, response.status_code)
        self.assertEqual(18, len(response.data))
        self.assertEqual('regression', response.data[0]['type'])
        self.assertEqual('simpleIndex', response.data[0]['config']['encoding'])
        self.assertEqual('noCluster', response.data[0]['config']['clustering'])
        self.assertEqual('linear', response.data[0]['config']['method'])
        self.assertEqual(123, response.data[0]['config']['random'])
        self.assertEqual(1, response.data[0]['config']['prefix_length'])
        self.assertEqual('no_padding', response.data[0]['config']['padding'])
        self.assertEqual('created', response.data[0]['status'])
        self.assertEqual(1, response.data[0]['split']['id'])

        self.assertEqual(3, response.data[17]['config']['prefix_length'])


class MethodConfiguration(TestCase):

    def job_obj(self):
        config = dict()
        config['encodings'] = ['simpleIndex']
        config['clusterings'] = ['noCluster']
        config['methods'] = ['randomForest']
        config['regression.randomForest'] = {'n_estimators': 15}
        config['regression.lasso'] = {'n_estimators': 15}
        config['prefix'] = {'prefix_length': 3, 'type': 'up_to', 'padding': 'no_padding'}
        obj = dict()
        obj['type'] = 'regression'
        obj['config'] = config
        obj['split_id'] = 1
        return obj

    def test_regression_random_forest(self):
        job = self.job_obj()

        config = create_config(job, 'simpleIndex', 'noCluster', 'randomForest', 3)

        self.assertEquals(False, 'regression.lasso' in config)
        self.assertDictEqual(config['regression.randomForest'], {
            'n_estimators': 15,
            'max_features': 'auto',
            'max_depth': None,
            'random_state': 21
        })

    def test_adds_conf_if_missing(self):
        job = self.job_obj()
        del job['config']['regression.randomForest']

        config = create_config(job, 'simpleIndex', 'noCluster', 'randomForest', 3)

        self.assertEquals(False, 'regression.lasso' in config)
        self.assertDictEqual(config['regression.randomForest'], {
            'n_estimators': 10,
            'max_features': 'auto',
            'max_depth': None,
            'random_state': 21
        })
