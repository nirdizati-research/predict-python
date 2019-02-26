"""
time series prediction tests
"""

import itertools

from django.test import TestCase

from core.constants import KNN, NO_CLUSTER, RNN, TIME_SERIES_PREDICTION, TIME_SERIES_PREDICTION_METHODS
from core.core import calculate
from core.tests.common import split_double, add_default_config, HidePrints
from encoders.encoding_container import EncodingContainer, ZERO_PADDING, SIMPLE_INDEX, \
    TIME_SERIES_PREDICTION_ENCODINGS, TIME_SERIES_PREDICTION_PADDINGS
from encoders.label_container import LabelContainer, ATTRIBUTE_STRING, THRESHOLD_CUSTOM, DURATION, \
    THRESHOLD_MEAN, TIME_SERIES_PREDICTION_LABELS


class TestTimeSeriesPrediction(TestCase):
    @staticmethod
    def get_job(method=RNN, encoding_method=SIMPLE_INDEX, padding=ZERO_PADDING, label=DURATION,
                add_elapsed_time=False):
        json = dict()
        json['clustering'] = NO_CLUSTER
        json['split'] = split_double()
        json['method'] = method
        json['encoding'] = EncodingContainer(encoding_method, padding=padding, prefix_length=4)
        json['incremental_train'] = {'base_model': None}
        if label == ATTRIBUTE_STRING:
            json['label'] = LabelContainer(label, attribute_name='creator')
        elif label == THRESHOLD_CUSTOM:
            json['label'] = LabelContainer(threshold_type=label, threshold=50)
        elif label == THRESHOLD_MEAN:
            json['label'] = LabelContainer(threshold_type=label, threshold=50)
        else:
            json['label'] = LabelContainer(label)
        json['add_elapsed_time'] = add_elapsed_time
        json['type'] = TIME_SERIES_PREDICTION

        if method != KNN:
            add_default_config(json)
        else:
            json['classification.knn'] = {'n_neighbors': 3}
        return json

    def test_no_exceptions(self):
        choices = [TIME_SERIES_PREDICTION_ENCODINGS, TIME_SERIES_PREDICTION_PADDINGS, TIME_SERIES_PREDICTION_METHODS,
                   TIME_SERIES_PREDICTION_LABELS]

        job_combinations = list(itertools.product(*choices))

        for (encoding, padding, method, label) in job_combinations:
            print(encoding, padding, method, label)

            job = self.get_job(method=method, encoding_method=encoding, padding=padding, label=label)
            with HidePrints():
                calculate(job)
