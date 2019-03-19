"""
Neural Networks tests
"""

from django.test import TestCase
from numpy import ndarray
from pandas import DataFrame

from src.encoding.complex_last_payload import complex
from src.encoding.models import ValueEncodings
from src.encoding.simple_index import simple_index
from src.labelling.models import LabelTypes, ThresholdTypes
from src.predictive_model.time_series_prediction.custom_time_series_prediction_models import RNNTimeSeriesPredictor
from src.utils.event_attributes import unique_events, get_additional_columns
from src.utils.file_service import get_log
from src.utils.tests_utils import create_test_encoding, create_test_labelling, create_test_log, \
    general_example_train_filename, \
    general_example_train_filepath, general_example_test_filename, general_example_test_filepath


class TestRNNTimeSeriesPredictor(TestCase):
    def setUp(self):
        self.train_log = get_log(create_test_log(log_name=general_example_train_filename,
                                                 log_path=general_example_train_filepath))
        self.train_event_names = unique_events(self.train_log)
        self.train_add_col = get_additional_columns(self.train_log)

        self.test_log = get_log(create_test_log(log_name=general_example_test_filename,
                                                log_path=general_example_test_filepath))
        self.test_event_names = unique_events(self.test_log)
        self.test_add_col = get_additional_columns(self.test_log)

    @staticmethod
    def _get_rnn_default_config(encoding=ValueEncodings.SIMPLE_INDEX.value):
        config = dict()
        config['n_units'] = 16
        config['rnn_type'] = 'lstm'
        config['n_epochs'] = 1
        config['encoding'] = encoding
        config['incremental_train'] = {'base_model': None}
        return config

    @staticmethod
    def _drop_columns_and_split(df: DataFrame) -> (DataFrame, ndarray):
        return df.drop(['trace_id', 'label'], 1), df['label']

    def test_rnn_time_series_predictor_simple_index_no_exceptions(self):
        encoding = create_test_encoding(value_encoding=ValueEncodings.SIMPLE_INDEX.value, prefix_length=5, padding=True)
        labelling = create_test_labelling(label_type=LabelTypes.DURATION.value,
                                          threshold_type=ThresholdTypes.THRESHOLD_MEAN.value)

        train_df = simple_index(self.train_log, labelling, encoding)
        test_df = simple_index(self.test_log, labelling, encoding)

        train_df, targets_df = self._drop_columns_and_split(train_df)

        test_df, _ = self._drop_columns_and_split(test_df)

        config = self._get_rnn_default_config()
        rnn_time_series_predictor = RNNTimeSeriesPredictor(**config)

        # with HidePrints():
        rnn_time_series_predictor.fit(train_df)
        rnn_time_series_predictor.predict(test_df)

    def test_rnn_time_series_predictor_complex_no_exceptions(self):
        encoding = create_test_encoding(value_encoding=ValueEncodings.COMPLEX.value, prefix_length=5, padding=True)
        labelling = create_test_labelling(label_type=LabelTypes.DURATION.value,
                                          threshold_type=ThresholdTypes.THRESHOLD_MEAN.value)

        train_df = complex(self.train_log, labelling, encoding, self.train_add_col)
        test_df = complex(self.test_log, labelling, encoding, self.test_add_col)

        train_df, targets_df = self._drop_columns_and_split(train_df)

        test_df, _ = self._drop_columns_and_split(test_df)

        config = self._get_rnn_default_config(encoding=ValueEncodings.COMPLEX.value)
        rnn_time_series_predictor = RNNTimeSeriesPredictor(**config)

        # with HidePrints():
        rnn_time_series_predictor.fit(train_df)
        rnn_time_series_predictor.predict(test_df)
