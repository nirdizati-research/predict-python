
from django.test import TestCase

from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModels
from src.utils.tests_utils import create_test_job, create_test_predictive_model


class TestCommon(TestCase):
    def test_str(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                prediction_method=ClassificationMethods.RANDOM_FOREST.value
            )
        )

        self.assertEqual(len(job.__str__()), len("{created_date: 2019-10-01 09:38:35.245361+00:00, modified_date: 2019-10-01 09:38:35.245655+00:00, error: , status: created, type: prediction, create_models: False, split: {'id': 1, 'type': 'single', 'test_size': 0.2, 'splitting_method': 'sequential', 'original_log_path': 'cache/log_cache/test_logs/general_example.xes'}, encoding: {'data_encoding': 'label_encoder', 'value_encoding': 'simpleIndex', 'add_elapsed_time': False, 'add_remaining_time': False, 'add_executed_events': False, 'add_resources_used': False, 'add_new_traces': False, 'features': {}, 'prefix_length': 1, 'padding': False, 'task_generation_type': 'only'}, labelling: {'type': 'next_activity', 'attribute_name': None, 'threshold_type': 'threshold_mean', 'threshold': 0.0, 'results': {}}, clustering: {'clustering_method': 'noCluster'}, predictive_model: {'n_estimators': 10, 'max_depth': None, 'max_features': 'auto'}, evaluation: [None], hyperparameter_optimizer: [None], incremental_train: [None]}"))
