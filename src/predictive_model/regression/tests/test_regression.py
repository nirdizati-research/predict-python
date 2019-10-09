"""
regression tests
"""

import unittest

from django.test import TestCase

from src.clustering.models import ClusteringMethods
from src.core.core import calculate
from src.core.tests.test_utils import split_double, add_default_config
from src.encoding.encoding_container import EncodingContainer, ZERO_PADDING
from src.encoding.models import ValueEncodings
from src.labelling.label_container import LabelContainer
from src.labelling.models import LabelTypes
from src.predictive_model.models import PredictiveModels
from src.predictive_model.regression.models import RegressionMethods
from src.utils.tests_utils import create_test_predictive_model, create_test_labelling, create_test_clustering, \
    create_test_job


class TestRegression(TestCase):
    @staticmethod
    def get_job(method=RegressionMethods.LINEAR.value, encoding_method=ValueEncodings.SIMPLE_INDEX.value,
                padding=ZERO_PADDING, label=LabelTypes.REMAINING_TIME.value,
                add_elapsed_time=False):
        json = dict()
        json['clustering'] = ClusteringMethods.NO_CLUSTER.value
        json['split'] = split_double()
        json['method'] = method
        json['encoding'] = EncodingContainer(encoding_method, padding=padding, prefix_length=4)
        json['labelling'] = LabelContainer(label)
        json['add_elapsed_time'] = add_elapsed_time
        json['type'] = PredictiveModels.REGRESSION.value
        json['incremental_train'] = {'base_model': None}

        add_default_config(json)
        return json

    @unittest.skip('needs refactoring')
    def test_no_exceptions(self):
        # filtered_labels = [x for x in REGRESSION_LABELS if
        #                    x != ATTRIBUTE_NUMBER]
        # # TODO: check how to add TRACE_NUMBER_ATTRIBUTE (test logs don't have numeric attributes)
        # choices = [ENCODING_METHODS, PADDINGS, REGRESSION_METHODS, filtered_labels]
        #
        # job_combinations = list(itertools.product(*choices))
        #
        # for (encoding, padding, method, label) in job_combinations:
        #     print(encoding, padding, method, label)
        #
        #     if method == 'nn' and padding == NO_PADDING:
        #         pass
        #
        #     job = self.get_job(method=method, encoding_method=encoding, padding=padding, label=label)
        #     with HidePrints():
        #         calculate(job)
        pass

    def test_regression_random_forest(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(predictive_model=PredictiveModels.REGRESSION.value,
                                                          prediction_method=RegressionMethods.RANDOM_FOREST.value),
            labelling=create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        print(result)
        self.assertDictEqual(result, {'mae': 0.0, 'mape': -1, 'rmse': 0.0, 'rscore': 1.0})

    def test_regression_lasso(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(predictive_model=PredictiveModels.REGRESSION.value,
                                                          prediction_method=RegressionMethods.LASSO.value),
            labelling=create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        print(result)
        self.assertDictEqual(result, {'mae': 0.0, 'mape': -1, 'rmse': 0.0, 'rscore': 1.0})

    def test_regression_linear(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(predictive_model=PredictiveModels.REGRESSION.value,
                                                          prediction_method=RegressionMethods.LINEAR.value),
            labelling=create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        print(result)
        self.assertDictEqual(result, {'mae': 0.0, 'mape': -1, 'rmse': 0.0, 'rscore': 1.0})

    @unittest.skip('needs refactoring')
    def test_regression_xgboost(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(predictive_model=PredictiveModels.REGRESSION.value,
                                                          prediction_method=RegressionMethods.XGBOOST.value),
            labelling=create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        print(result)
        self.assertDictEqual(result, {'mae': 0.00011968612670898438, 'mape': -1, 'rmse': 0.00011968612670898438,
                                      'rscore': 0.0})

    def test_regression_nn(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(predictive_model=PredictiveModels.REGRESSION.value,
                                                          prediction_method=RegressionMethods.NN.value),
            labelling=create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        print(result)
        self.assertDictEqual(result, {'mae': 0.0, 'mape': -1, 'rmse': 0.0, 'rscore': 1.0})
