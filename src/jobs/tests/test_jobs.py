from django.test import TestCase
from django_rq.queues import get_queue
from rest_framework import status
from rest_framework.test import APITestCase, APIClient

from src.clustering.models import ClusteringMethods
from src.encoding.models import ValueEncodings, TaskGenerationTypes
from src.hyperparameter_optimization.models import HyperOptLosses, HyperparameterOptimizationMethods
from src.jobs.models import Job, JobStatuses, JobTypes
from src.jobs.tasks import prediction_task
from src.labelling.models import ThresholdTypes, LabelTypes
from src.predictive_model.classification.methods_default_config import classification_random_forest
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModels
from src.predictive_model.regression.models import RegressionMethods
from src.split.models import SplitTypes
from src.utils.tests_utils import general_example_filepath, create_test_job, create_test_log, general_example_filename, \
    create_test_split, create_test_predictive_model, create_test_hyperparameter_optimizer, create_test_clustering, \
    create_test_encoding, create_test_labelling


class JobModelTest(TestCase):
    def setUp(self):
        create_test_job()
        create_test_job(job_type='asdf')
        Job.objects.create(type=JobTypes.PREDICTION.value, split=create_test_split(), encoding=None, labelling=None)

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
        self.assertEqual(None, job.evaluation)
        self.assertEqual("ValueError('Type asdf not supported',)", job.error)

    def test_missing_attributes(self):
        self.assertRaises(AttributeError, prediction_task, 3)
        job = Job.objects.get(pk=3)

        self.assertEqual('error', job.status)
        self.assertEqual(None, job.evaluation)
        self.assertEqual("AttributeError(\"'NoneType' object has no attribute 'type'\",)", job.error)


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
        self.assertFalse(classification_random_forest() ==
                         job.predictive_model.classification
                         .__getattribute__(ClassificationMethods.RANDOM_FOREST.value.lower()).to_dict())


class CreateJobsTests(APITestCase):
    def setUp(self):
        log = create_test_log(log_name=general_example_filepath, log_path=general_example_filepath)
        create_test_split(split_type=SplitTypes.SPLIT_SINGLE.value, original_log=log)

    def tearDown(self):
        get_queue().empty()

    @staticmethod
    def job_obj():
        return {
            'type': PredictiveModels.CLASSIFICATION.value,
            'split_id': 1,
            'config': {
                'encodings': [ValueEncodings.SIMPLE_INDEX.value],
                'clusterings': [ClusteringMethods.NO_CLUSTER.value],
                'methods': ['knn'],
                'labelling': {
                    'type': LabelTypes.REMAINING_TIME.value,
                    'attribute_name': None,
                    'threshold_type': ThresholdTypes.THRESHOLD_MEAN.value,
                    'threshold': 0,
                    'add_remaining_time': False,
                    'add_elapsed_time': False
                },
                'random': 123,
                'kmeans': {},
                'encoding': {
                    'prefix_length': 3,
                    'generation_type': 'only',
                    'padding': 'zero_padding'
                }
            }
        }

    def test_class_job_creation(self):
        client = APIClient()
        response = client.post('/jobs/multiple', self.job_obj(), format='json')

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]['type'], 'prediction')
        self.assertEqual(response.data[0]['config']['predictive_model']['predictive_model'], 'classification')
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
        self.assertEqual(response.data[0]['config']['predictive_model']['prediction_method'],
                         ClassificationMethods.KNN.value)
        self.assertFalse('kmeans' in response.data[0]['config'])
        self.assertDictEqual(response.data[0]['config']['labelling'],
                         {'type': 'remaining_time', 'attribute_name': None,
                          'threshold_type': ThresholdTypes.THRESHOLD_MEAN.value,
                          'threshold': 0})
        self.assertEqual(response.data[0]['status'], 'created')

    @staticmethod
    def job_obj2():
        return {
            'type': 'regression',
            'split_id': 1,
            'config': {
                'encodings': [ValueEncodings.SIMPLE_INDEX.value, ValueEncodings.BOOLEAN.value,
                              ValueEncodings.COMPLEX.value],
                'clusterings': [ClusteringMethods.KMEANS.value],
                'methods': [RegressionMethods.LINEAR.value, RegressionMethods.LASSO.value],
                'random': 123,
                'kmeans': {'max_iter': 100},
                'encoding': {'prefix_length': 3, 'generation_type': TaskGenerationTypes.UP_TO.value,
                             'padding': 'no_padding'},
                'labelling': {'type': LabelTypes.REMAINING_TIME.value}
            }
        }

    def test_reg_job_creation(self):
        client = APIClient()
        response = client.post('/jobs/multiple', self.job_obj2(), format='json')

        self.assertEqual(status.HTTP_201_CREATED, response.status_code)
        self.assertEqual(18, len(response.data))
        self.assertEqual(response.data[0]['type'], 'prediction')
        self.assertEqual(response.data[0]['config']['predictive_model']['predictive_model'], 'regression')
        self.assertEqual(ValueEncodings.SIMPLE_INDEX.value, response.data[0]['config']['encoding']['value_encoding'])
        self.assertEqual(ClusteringMethods.KMEANS.value, response.data[0]['config']['clustering']['clustering_method'])
        self.assertEqual(RegressionMethods.LINEAR.value,
                         response.data[0]['config']['predictive_model']['prediction_method'])
        self.assertEqual(1, response.data[0]['config']['encoding']['prefix_length'])
        self.assertEqual(False, response.data[0]['config']['encoding']['padding'])
        self.assertEqual(100, response.data[0]['config']['clustering']['max_iter'])
        self.assertEqual(JobStatuses.CREATED.value, response.data[0]['status'])
        self.assertEqual(1, response.data[0]['config']['split']['id'])

        self.assertEqual(3, response.data[17]['config']['encoding']['prefix_length'])

    @staticmethod
    def job_label():
        return{
            'type': 'labelling',
            'split_id': 1,
            'config': {
                'labelling': {
                    'type': 'remaining_time',
                    'attribute_name': None,
                    'threshold_type': ThresholdTypes.THRESHOLD_MEAN.value,
                    'threshold': 0,
                    'add_remaining_time': False,
                    'add_elapsed_time': False
                },
                'encoding': {
                    'prefix_length': 3,
                    'generation_type': 'only',
                    'padding': 'zero_padding'
                }
            }
        }

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
        return {
            'type': 'regression',
            'split_id': 1,
            'config': {
                'encodings': ['simpleIndex'],
                'clusterings': ['noClustering'],
                'methods': ['randomForest'],
                'regression.:randomForest': {'n_estimators': 15},
                'regression.:lasso': {'n_estimators': 15},
                'encoding': {
                    'prefix_length': 3,
                    'generation_type': 'up_to',
                    'padding': 'no_padding'
                }
            }
        }

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
