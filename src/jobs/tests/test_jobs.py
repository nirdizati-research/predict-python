import unittest
from pprint import pprint

from django.test import TestCase
from django_rq.queues import get_queue
from rest_framework import status
from rest_framework.test import APITestCase, APIClient

from src.clustering.models import ClusteringMethods
from src.core.tests.common import add_default_config
from src.encoding.models import ValueEncodings
from src.hyperparameter_optimization.models import HyperOptLosses, HyperparameterOptimizationMethods
from src.jobs.models import Job, JobStatuses, JobTypes
from src.jobs.tasks import prediction_task
from src.labelling.models import ThresholdTypes, LabelTypes
from src.logs.models import Log
from src.predictive_model.classification.methods_default_config import classification_random_forest
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModels
from src.split.models import Split, SplitTypes
from src.utils.tests_utils import general_example_filepath, create_test_job, create_test_log, general_example_filename, \
    create_test_split, create_test_predictive_model, create_test_hyperparameter_optimizer, create_test_clustering, \
    create_test_encoding, create_test_labelling


class JobModelTest(TestCase):
    def setUp(self):
        create_test_job()

    def test_default(self):
        job = Job.objects.get(pk=1)

        self.assertEqual('created', job.status)
        self.assertIsNotNone(job.created_date)
        self.assertIsNotNone(job.modified_date)
        self.assertIsNone(job.evaluation)

    def test_modified(self):
        job = Job.objects.get(pk=1)
        job.status = JobStatuses.COMPLETED.value

        self.assertNotEquals(job.created_date, job.modified_date)

    def test_to_dict(self):
        job = Job.objects.get(pk=1).to_dict()

        self.assertEquals(JobTypes.PREDICTION.value, job['type'])
        self.assertEquals(PredictiveModels.CLASSIFICATION.value, job['predictive_model']['predictive_model'])
        self.assertDictEqual({'type': 'single',
                              'original_log_path': general_example_filepath,
                              'splitting_method': 'sequential',
                              'test_size': 0.2,
                              'id': 1},
                             job['split'])
        self.assertEquals(job['labelling'], {
            'attribute_name': None,
            'threshold': 0,
            'threshold_type': 'threshold_mean',
            'type': 'next_activity'
        })

    def test_prediction_task(self):
        prediction_task(1)

        job = Job.objects.get(pk=1)

        self.assertEqual('completed', job.status)
        self.assertNotEqual({}, job.evaluation)

    def test_create_models_config_missing(self):
        job = Job.objects.get(pk=1)
        del job.create_models  # TODO fixme should we add this field?
        job.save()
        prediction_task(1)

        job = Job.objects.get(pk=1)

        self.assertEqual('completed', job.status)
        self.assertNotEqual({}, job.evaluation)

    def test_prediction_task_error(self):
        self.assertRaises(ValueError, prediction_task, 2)
        job = Job.objects.get(pk=2)

        self.assertEqual('error', job.status)
        self.assertEqual({}, job.result)
        self.assertEqual("ValueError('Type not supported', 'asdsd')", job.error)

    def test_missing_attributes(self):
        self.assertRaises(KeyError, prediction_task, 3)
        job = Job.objects.get(pk=3)

        self.assertEqual('error', job.status)
        self.assertEqual({}, job.result)
        self.assertEqual("KeyError('label',)", job.error)


class Hyperopt(TestCase):

    def test_hyperopt(self):
        job = Job.objects.create(
            split=create_test_split(
                split_type=SplitTypes.SPLIT_SINGLE.value,
                original_log=create_test_log(log_name=general_example_filename, log_path=general_example_filepath)
            ),
            encoding=create_test_encoding(
                value_encoding=ValueEncodings.SIMPLE_INDEX.value,
                prefix_length=3,
                padding=False
            ),
            labelling=create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value),
            clustering=create_test_clustering(
                clustering_type=ClusteringMethods.KMEANS.value
            ),
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                prediction_method=ClassificationMethods.RANDOM_FOREST.value
            ),
            hyperparameter_optimizer=create_test_hyperparameter_optimizer(
                hyperoptim_type=HyperparameterOptimizationMethods.HYPEROPT.value,
                performance_metric=HyperOptLosses.ACC.value,
                max_evals=2
            )
        )
        prediction_task(job.pk)
        job = Job.objects.get(pk=1)
        self.assertFalse(classification_random_forest() == job.config['classification.randomForest'])


class CreateJobsTests(APITestCase):
    def setUp(self):
        log = create_test_log(log_name=general_example_filepath, log_path=general_example_filepath)
        create_test_split(split_type=SplitTypes.SPLIT_SINGLE.value, original_log=log)

    def tearDown(self):
        get_queue().empty()

    @staticmethod
    def job_obj():
        config = dict()
        config['encodings'] = ['simpleIndex']
        config['clusterings'] = ['noCluster']
        config['methods'] = ['knn']
        config['label'] = {'type': 'remaining_time', 'attribute_name': None,
                           'threshold_type': ThresholdTypes.THRESHOLD_MEAN.value,
                           'threshold': 0, 'add_remaining_time': False, 'add_elapsed_time': False}
        config['random'] = 123
        config['kmeans'] = {}
        config['encoding'] = {'prefix_length': 3, 'generation_type': 'only', 'padding': 'zero_padding'}
        obj = dict()
        obj['type'] = PredictiveModels.CLASSIFICATION.value
        obj['config'] = config
        obj['split_id'] = 1
        return obj

    def test_class_job_creation(self):
        client = APIClient()
        response = client.post('/jobs/multiple', self.job_obj(), format='json')
        pprint(response.data[0])
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['type'], 'classification')
        self.assertDictEqual(response.data[0]['config']['encoding'], {
            'prefix_length': 3,
            'task_generation_type': 'only',
            'value_encoding': 'simpleIndex',
            'add_elapsed_time': False,
            'add_executed_events': False,
            'add_new_traces': False,
            'add_remaining_time': False,
            'add_resources_used': False,
            'data_encoding': 'label_encoder',
            'features': {},
            'padding': True,
        })
        self.assertEqual(response.data[0]['config']['clustering'], {
            'clustering_method': ClusteringMethods.NO_CLUSTER.value
        })
        self.assertEqual(response.data[0]['config']['predictive_model']['classification_method'],
                         ClassificationMethods.KNN.value)
        self.assertFalse('kmeans' in response.data[0]['config'])
        self.assertDictEqual(response.data[0]['config']['labelling'],
                         {'type': 'remaining_time', 'attribute_name': 'label',
                          'threshold_type': ThresholdTypes.THRESHOLD_MEAN.value,
                          'threshold': 0})
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
        config['label'] = {'type': 'remaining_time', 'attribute_name': None,
                           'threshold_type': ThresholdTypes.THRESHOLD_MEAN.value,
                           'threshold': 0, 'add_remaining_time': False, 'add_elapsed_time': False}
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
        self.assertEqual(response.data[0]['config']['encoding']['value_encoding'], 'simpleIndex')
        self.assertEqual(response.data[0]['config']['encoding']['prefix_length'], 3)
        self.assertEqual(response.data[0]['config']['labelling'],
                         {'type': 'remaining_time', 'attribute_name': None,
                          'threshold_type': ThresholdTypes.THRESHOLD_MEAN.value,
                          'threshold': 0})
        self.assertEqual(response.data[0]['config']['encoding']['padding'], True)
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

    # def test_regression_random_forest(self):
    #     job = self.job_obj()
    #
    #     config = create_config(job, 'simpleIndex', 'noCluster', 'randomForest', 3)
    #
    #     self.assertEquals(False, 'regression.lasso' in config)
    #     self.assertDictEqual(config['regression.randomForest'], {
    #         'n_estimators': 15,
    #         'max_features': 'auto',
    #         'max_depth': None,
    #         'n_jobs': -1,
    #         'random_state': 21
    #     })

    # def test_adds_conf_if_missing(self):
    #     job = self.job_obj()
    #     del job['config']['regression.randomForest']
    #
    #     config = create_config(job, 'simpleIndex', 'noCluster', 'randomForest', 3)
    #
    #     self.assertEquals(False, 'regression.lasso' in config)
    #     self.assertDictEqual(config['regression.randomForest'], {
    #         'n_estimators': 10,
    #         'max_features': 'auto',
    #         'max_depth': None,
    #         'n_jobs': -1,
    #         'random_state': 21
    #     })
