"""
hyperopt tests
"""

from django.test import TestCase

from core.hyperopt_wrapper import calculate_hyperopt
from core.tests.common import add_default_config, repair_example
from encoders.encoding_container import EncodingContainer, ZERO_PADDING
from encoders.label_container import LabelContainer


class TestHyperopt(TestCase):
    """Proof of concept tests"""

    @staticmethod
    def get_job():
        json = dict()
        json["split"] = repair_example()
        json["method"] = "randomForest"
        json["encoding"] = EncodingContainer(prefix_length=8, padding=ZERO_PADDING)
        json["type"] = "classification"
        json['clustering'] = 'noCluster'
        json['label'] = LabelContainer(add_elapsed_time=True)
        json['hyperopt'] = {'use_hyperopt': True, 'max_evals': 2, 'performance_metric': 'acc'}
        return json

    def test_class_randomForest(self):
        job = self.get_job()
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_class_knn(self):
        job = self.get_job()
        job["method"] = "knn"
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_class_xgboost(self):
        job = self.get_job()
        job["method"] = "xgboost"
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_class_decision_tree(self):
        job = self.get_job()
        job["method"] = "decisionTree"
        job['classification.decisionTree'] = {}
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_regression_random_forest(self):
        job = self.get_job()
        job["type"] = "regression"
        job['hyperopt']['performance_metric'] = 'rmse'
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_regression_linear(self):
        job = self.get_job()
        job["type"] = "regression"
        job["method"] = "linear"
        job['hyperopt']['performance_metric'] = 'rmse'
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_regression_lasso(self):
        job = self.get_job()
        job["type"] = "regression"
        job["method"] = "lasso"
        job['hyperopt']['performance_metric'] = 'rmse'
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)

    def test_regression_xgboost(self):
        job = self.get_job()
        job["type"] = "regression"
        job["method"] = "xgboost"
        job['hyperopt']['performance_metric'] = 'rmse'
        add_default_config(job)
        results, config, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
        self.assertIsNotNone(config)
