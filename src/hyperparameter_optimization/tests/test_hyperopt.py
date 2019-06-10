"""
hyperopt tests
"""
import unittest

from django.test import TestCase

from src.hyperparameter_optimization.hyperopt_wrapper import calculate_hyperopt
from src.hyperparameter_optimization.models import HyperOptLosses
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModels
from src.predictive_model.regression.models import RegressionMethods
from src.utils.tests_utils import create_test_predictive_model, create_test_job, create_test_hyperparameter_optimizer, \
    create_test_encoding


class TestHyperopt(TestCase):
    """Proof of concept tests"""

    @staticmethod
    def get_job(predictive_model: str, prediction_method: str, metric: HyperOptLosses = HyperOptLosses.ACC.value):
        encoding = create_test_encoding(prefix_length=8, padding=True)
        pred_model = create_test_predictive_model(predictive_model=predictive_model,
                                                  prediction_method=prediction_method)
        hyperparameter_optimizer = create_test_hyperparameter_optimizer(performance_metric=metric)

        job = create_test_job(predictive_model=pred_model,
                              encoding=encoding,
                              hyperparameter_optimizer=hyperparameter_optimizer)
        return job

    def test_class_randomForest(self):
        job = self.get_job(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.RANDOM_FOREST.value)
        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    # def test_class_knn(self):
    #     job = self.get_job(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.KNN.value)
    #
    #     results, _ = calculate_hyperopt(job)
    #     self.assertIsNotNone(results)

    def test_class_xgboost(self):
        job = self.get_job(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.XGBOOST.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    def test_class_decision_tree(self):
        job = self.get_job(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.DECISION_TREE.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    @unittest.skip
    def test_regression_random_forest(self):
        job = self.get_job(PredictiveModels.REGRESSION.value, RegressionMethods.RANDOM_FOREST.value,
                           HyperOptLosses.RMSE.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    def test_regression_linear(self):
        job = self.get_job(PredictiveModels.REGRESSION.value, RegressionMethods.LINEAR.value,
                           HyperOptLosses.RMSE.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    def test_regression_lasso(self):
        job = self.get_job(PredictiveModels.REGRESSION.value, RegressionMethods.LASSO.value,
                           HyperOptLosses.RMSE.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)

    def test_regression_xgboost(self):
        job = self.get_job(PredictiveModels.REGRESSION.value, RegressionMethods.XGBOOST.value,
                           HyperOptLosses.RMSE.value)

        results, _, _ = calculate_hyperopt(job)
        self.assertIsNotNone(results)
