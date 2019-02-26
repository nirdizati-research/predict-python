"""
Neural Networks tests
"""

import numpy as np
import pandas as pd
from django.test import TestCase

from core.nn_models import NNClassifier, NNRegressor, RNNTimeSeriesPredictor
from core.tests.common import HidePrints
from encoders.encoding_container import SIMPLE_INDEX


class TestNNModels(TestCase):
    @staticmethod
    def _get_nn_default_config(encoding=SIMPLE_INDEX, binary=False):
        config = dict()
        config['n_hidden_layers'] = 2
        config['n_hidden_units'] = 10
        config['activation'] = 'relu'
        config['n_epochs'] = 1
        config['encoding'] = encoding
        config['dropout_rate'] = 0.1
        config['is_binary_classifier'] = binary
        return config

    def test_nn_classifier_simple_index_binary_no_exceptions(self):
        config = self._get_nn_default_config(binary=True)

        nn_classifier = NNClassifier(**config)

        train_df = pd.DataFrame(np.ones((2, 3), dtype=np.int))
        test_df = pd.DataFrame(np.ones((2, 3), dtype=np.int))
        targets_df = np.ones((2, 1), dtype=np.int) * 1

        with HidePrints():
            nn_classifier.fit(train_df, targets_df)
            nn_classifier.predict(test_df)
            nn_classifier.predict_proba(test_df)

    def test_nn_classifier_simple_index_multiclass_no_exceptions(self):
        config = self._get_nn_default_config()

        nn_classifier = NNClassifier(**config)

        train_df = pd.DataFrame(np.ones((2, 3), dtype=np.int))
        test_df = pd.DataFrame(np.ones((2, 3), dtype=np.int))
        targets_df = np.ones((2, 1), dtype=np.int) * 5

        with HidePrints():
            nn_classifier.fit(train_df, targets_df)
            nn_classifier.predict(test_df)
            nn_classifier.predict_proba(test_df)

    def test_nn_regressor_simple_index_no_exceptions(self):
        config = self._get_nn_default_config()

        nn_regressor = NNRegressor(**config)

        train_df = pd.DataFrame(np.ones((2, 3), dtype=np.int))
        test_df = pd.DataFrame(np.ones((2, 3), dtype=np.int))
        targets_df = np.ones((2, 1)) * 0.5

        with HidePrints():
            nn_regressor.fit(train_df, targets_df)
            nn_regressor.predict(test_df)


class TestRNNModels(TestCase):
    @staticmethod
    def _get_rnn_default_config(encoding=SIMPLE_INDEX):
        config = dict()
        config['n_units'] = 16
        config['rnn_type'] = 'lstm'
        config['n_epochs'] = 1
        config['encoding'] = encoding
        return config

    def test_rnn_time_series_predictor_simple_index_no_exceptions(self):
        config = self._get_rnn_default_config()

        rnn_time_series_predictor = RNNTimeSeriesPredictor(**config)

        train_df = pd.DataFrame(np.ones((2, 3), dtype=np.int))
        test_df = pd.DataFrame(np.ones((2, 3), dtype=np.int))

        with HidePrints():
            rnn_time_series_predictor.fit(train_df)
            rnn_time_series_predictor.predict(test_df)
