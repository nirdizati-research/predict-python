"""
hyperopt tests
"""

from django.test import TestCase

from src.core.tests.common import add_default_config, repair_example
from src.encoding.encoding_container import EncodingContainer, ZERO_PADDING
from src.hyperparameter_optimization.hyperopt_wrapper import calculate_hyperopt
from src.labelling.label_container import LabelContainer
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictionTypes
from src.utils.tests_utils import create_test_predictive_model, create_test_job, create_test_hyperparameter_optimizer, \
    create_test_encoding


class TestHyperopt(TestCase):
    """Proof of concept tests"""

    @staticmethod
    def get_job():
        json = dict()
        json['split'] = repair_example()
        json['method'] = 'randomForest'
        json['encoding'] = EncodingContainer(prefix_length=8, padding=ZERO_PADDING)
        json['type'] = 'classification'
        json['clustering'] = 'noCluster'
        json['label'] = LabelContainer(add_elapsed_time=True)
        json['hyperopt'] = {'use_hyperopt': True, 'max_evals': 2, 'performance_metric': 'acc'}
        json['incremental_train'] = {'base_model': None}
        return json

    def test_class_randomForest(self):
        encoding = create_test_encoding(prefix_length=8, padding=True)
        predictive_model = create_test_predictive_model(predictive_model=PredictionTypes.CLASSIFICATION.value,
                                                        prediction_method=ClassificationMethods.RANDOM_FOREST.value)
        hyperparameter_optimizer = create_test_hyperparameter_optimizer()

        job = create_test_job(predictive_model=predictive_model,
                              encoding=encoding,
                              hyperparameter_optimizer=hyperparameter_optimizer)

        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_class_knn(self):
        job = self.get_job()
        job['method'] = 'knn'
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_class_xgboost(self):
        job = self.get_job()
        job['method'] = 'xgboost'
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_class_decision_tree(self):
        job = self.get_job()
        job['method'] = 'decisionTree'
        job['classification.decisionTree'] = {}
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_regression_random_forest(self):
        job = self.get_job()
        job['type'] = 'regression'
        job['hyperopt']['performance_metric'] = 'rmse'
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_regression_linear(self):
        job = self.get_job()
        job['type'] = 'regression'
        job['method'] = 'linear'
        job['hyperopt']['performance_metric'] = 'rmse'
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_regression_lasso(self):
        job = self.get_job()
        job['type'] = 'regression'
        job['method'] = 'lasso'
        job['hyperopt']['performance_metric'] = 'rmse'
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_regression_xgboost(self):
        job = self.get_job()
        job['type'] = 'regression'
        job['method'] = 'xgboost'
        job['hyperopt']['performance_metric'] = 'rmse'
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)
