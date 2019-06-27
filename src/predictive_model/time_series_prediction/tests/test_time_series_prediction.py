"""
time series prediction tests
"""

import unittest

from django.test import TestCase

from src.clustering.models import ClusteringMethods
from src.core.core import calculate
from src.core.tests.common import split_double, add_default_config
from src.encoding.encoding_container import EncodingContainer, ZERO_PADDING
from src.encoding.models import ValueEncodings
from src.labelling.label_container import LabelContainer
from src.labelling.models import LabelTypes, ThresholdTypes
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.models import PredictiveModels
from src.predictive_model.time_series_prediction.models import TimeSeriesPredictionMethods
from src.utils.tests_utils import create_test_job, create_test_predictive_model, create_test_labelling, \
    create_test_clustering, create_test_encoding


class TestTimeSeriesPrediction(TestCase):
    @staticmethod
    def get_job(method=TimeSeriesPredictionMethods.RNN.value, encoding_method=ValueEncodings.SIMPLE_INDEX.value,
                padding=ZERO_PADDING, label=LabelTypes.DURATION.value,
                add_elapsed_time=False):
        json = dict()
        json['clustering'] = ClusteringMethods.NO_CLUSTER.value
        json['split'] = split_double()
        json['method'] = method
        json['encoding'] = EncodingContainer(encoding_method, padding=padding, prefix_length=4)
        json['incremental_train'] = {'base_model': None}
        if label == LabelTypes.ATTRIBUTE_STRING.value:
            json['labelling'] = LabelContainer(label, attribute_name='creator')
        elif label == ThresholdTypes.THRESHOLD_CUSTOM.value:
            json['labelling'] = LabelContainer(threshold_type=label, threshold=50)
        elif label == ThresholdTypes.THRESHOLD_MEAN.value:
            json['labelling'] = LabelContainer(threshold_type=label, threshold=50)
        else:
            json['labelling'] = LabelContainer(label)
        json['add_elapsed_time'] = add_elapsed_time
        json['type'] = PredictiveModels.TIME_SERIES_PREDICTION.value

        if method != ClassificationMethods.KNN.value:
            add_default_config(json)
        else:
            json['classification.knn'] = {'n_neighbors': 3}
        return json

    @unittest.skip('needs refactoring')
    def test_no_exceptions(self):
        # choices = [TIME_SERIES_PREDICTION_ENCODINGS, TIME_SERIES_PREDICTION_PADDINGS, TIME_SERIES_PREDICTION_METHODS,
        #            TIME_SERIES_PREDICTION_LABELS]

        # job_combinations = list(itertools.product(*choices))

        # for (encoding, padding, method, label) in job_combinations:
        #     print(encoding, padding, method, label)
        #
        #     job = self.get_job(method=method, encoding_method=encoding, padding=padding, label=label)
        #     with HidePrints():
        #         calculate(job)
        pass

    def test_tsp_lstm(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.TIME_SERIES_PREDICTION.value,
                prediction_method=TimeSeriesPredictionMethods.RNN.value,
                configuration={'rnn_type': 'lstm'}),
            labelling=create_test_labelling(),
            encoding=create_test_encoding(prefix_length=2, padding=True),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'nlevenshtein': 0.6})

    def test_tsp_gru(self):
        job = create_test_job(
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.TIME_SERIES_PREDICTION.value,
                prediction_method=TimeSeriesPredictionMethods.RNN.value,
                configuration={'rnn_type': 'gru'}),
            labelling=create_test_labelling(),
            encoding=create_test_encoding(prefix_length=2, padding=True),
            clustering=create_test_clustering(clustering_type=ClusteringMethods.NO_CLUSTER.value)
        )
        result, _ = calculate(job)
        del result['elapsed_time']
        self.assertDictEqual(result, {'nlevenshtein': 0.6})
