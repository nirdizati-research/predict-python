from django.test import TestCase
from django_rq.queues import get_queue
from rest_framework import status
from rest_framework.test import APITestCase, APIClient

from core.constants import CLASSIFICATION, REGRESSION
from core.default_configuration import _classification_random_forest
from core.tests.common import add_default_config
from encoders.label_container import LabelContainer, THRESHOLD_MEAN
from jobs.job_creator import create_config
from jobs.models import Job
from jobs.tasks import prediction_task
from logs.models import Log, Split
from utils.tests_utils import general_example_filepath


class JobModelTest(TestCase):
    def setUp(self):
        self.config = {'key': 123,
                       'method': 'randomForest',
                       'encoding': {'method': 'simpleIndex', "prefix_length": 1,
                                    "padding": 'no_padding', 'generation_type': 'only'},
                       'clustering': 'noCluster',
                       "create_models": False,
                       "label": {'type': 'remaining_time'}
                       }
        log = Log.objects.create(name="general_example.xes", path=general_example_filepath)
        split = Split.objects.create(original_log=log)
        Job.objects.create(config=add_default_config(self.config, prediction_method=CLASSIFICATION), split=split,
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
                              'original_log_path': general_example_filepath,
                              'config': {},
                              'id': 1},
                             job['split'])
        self.assertEquals(123, job['key'])
        self.assertEquals(job['label'], LabelContainer())

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
        self.assertEqual("KeyError('label',)", job.error)


class Hyperopt(TestCase):
    def setUp(self):
        self.config = {
            'method': 'randomForest',
            'encoding': {'method': 'simpleIndex', "prefix_length": 3, "padding": 'no_padding'},
            'clustering': 'noCluster',
            "create_models": False,
            "label": {"type": "remaining_time"},
            "hyperopt": {"use_hyperopt": True, "max_evals": 2, "performance_metric": "acc"}
        }
        log = Log.objects.create(name="general_example.xes", path=general_example_filepath)
        split = Split.objects.create(original_log=log)
        Job.objects.create(config=add_default_config(self.config, prediction_method=CLASSIFICATION), split=split,
                           type=CLASSIFICATION)

    def test_hyperopt(self):
        prediction_task(1)
        job = Job.objects.get(id=1)
        self.assertFalse(_classification_random_forest() == job.config['classification.randomForest'])


class CreateJobsTests(APITestCase):
    def setUp(self):
        log = Log.objects.create(name="general_example.xes", path=general_example_filepath)
        Split.objects.create(original_log=log)

    def tearDown(self):
        get_queue().empty()

    @staticmethod
    def job_obj():
        config = dict()
        config['encodings'] = ['simpleIndex']
        config['clusterings'] = ['noCluster']
        config['methods'] = ['knn']
        config['label'] = {'type': 'remaining_time', "attribute_name": None, "threshold_type": THRESHOLD_MEAN,
                           "threshold": 0, "add_remaining_time": False, "add_elapsed_time": False}
        config['random'] = 123
        config['kmeans'] = {}
        config['encoding'] = {'prefix_length': 3, 'generation_type': 'only', 'padding': 'zero_padding'}
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
        self.assertEqual(response.data[0]['config']['encoding'],
                         {'method': 'simpleIndex', 'padding': 'zero_padding', 'prefix_length': 3,
                          'generation_type': 'only'})
        self.assertEqual(response.data[0]['config']['clustering'], 'noCluster')
        self.assertEqual(response.data[0]['config']['method'], 'knn')
        self.assertEqual(response.data[0]['config']['random'], 123)
        self.assertFalse('kmeans' in response.data[0]['config'])
        self.assertEqual(response.data[0]['config']['label'],
                         {'type': 'remaining_time', "attribute_name": None, "threshold_type": THRESHOLD_MEAN,
                          "threshold": 0, "add_remaining_time": False, "add_elapsed_time": False})
        self.assertEqual(response.data[0]['status'], 'created')

    @staticmethod
    def job_obj2():
        config = dict()
        config['encodings'] = ['simpleIndex', 'boolean', 'complex']
        config['clusterings'] = ['kmeans']
        config['methods'] = ['linear', 'lasso']
        config['random'] = 123
        config['kmeans'] = {'max_iter': 100}
        config['encoding'] = {'prefix_length': 3, 'generation_type': 'up_to', 'padding': 'no_padding'}
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
        self.assertEqual('simpleIndex', response.data[0]['config']['encoding']['method'])
        self.assertEqual('kmeans', response.data[0]['config']['clustering'])
        self.assertEqual('linear', response.data[0]['config']['method'])
        self.assertEqual(123, response.data[0]['config']['random'])
        self.assertEqual(1, response.data[0]['config']['encoding']['prefix_length'])
        self.assertEqual('no_padding', response.data[0]['config']['encoding']['padding'])
        self.assertEqual(100, response.data[0]['config']['kmeans']['max_iter'])
        self.assertEqual('created', response.data[0]['status'])
        self.assertEqual(1, response.data[0]['split_id'])

        self.assertEqual(3, response.data[17]['config']['encoding']['prefix_length'])

    @staticmethod
    def job_label():
        config = dict()
        config['label'] = {"type": 'remaining_time', "attribute_name": None, "threshold_type": THRESHOLD_MEAN,
                           "threshold": 0, "add_remaining_time": False, "add_elapsed_time": False}
        config['encoding'] = {'prefix_length': 3, 'generation_type': 'only', 'padding': 'zero_padding'}
        obj = dict()
        obj['type'] = 'labelling'
        obj['config'] = config
        obj['split_id'] = 1
        return obj

    def test_labelling_job_creation(self):
        client = APIClient()
        response = client.post('/jobs/multiple', self.job_label(), format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['type'], 'labelling')
        self.assertEqual(response.data[0]['config']['encoding']['method'], 'simpleIndex')
        self.assertEqual(response.data[0]['config']['encoding']['prefix_length'], 3)
        self.assertEqual(response.data[0]['config']['label'],
                         {'type': 'remaining_time', "attribute_name": None, "threshold_type": THRESHOLD_MEAN,
                          "threshold": 0, "add_remaining_time": False, "add_elapsed_time": False})
        self.assertEqual(response.data[0]['config']['encoding']['padding'], 'zero_padding')
        self.assertEqual(response.data[0]['status'], 'created')


class MethodConfiguration(TestCase):

    @staticmethod
    def job_obj():
        config = dict()
        config['encodings'] = ['simpleIndex']
        config['clusterings'] = ['noCluster']
        config['methods'] = ['randomForest']
        config['regression.randomForest'] = {'n_estimators': 15}
        config['regression.lasso'] = {'n_estimators': 15}
        config['encoding'] = {'prefix_length': 3, 'generation_type': 'up_to', 'padding': 'no_padding'}
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
            'n_jobs': -1,
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
            'n_jobs': -1,
            'random_state': 21
        })
