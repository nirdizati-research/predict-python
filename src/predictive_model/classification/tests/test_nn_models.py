"""
Neural Networks tests
"""

from django.test import TestCase
from numpy import ndarray
from pandas import DataFrame

from src.core.tests.test_utils import HidePrints
from src.encoding.models import ValueEncodings
from src.encoding.simple_index import simple_index
from src.labelling.models import LabelTypes, ThresholdTypes
from src.logs.log_service import get_log
from src.predictive_model.classification.custom_classification_models import NNClassifier
from src.utils.event_attributes import unique_events, get_additional_columns
from src.utils.tests_utils import create_test_encoding, create_test_labelling, create_test_log, \
    general_example_train_filename, \
    general_example_train_filepath, general_example_test_filename, general_example_test_filepath_xes


class TestNNClassifier(TestCase):
    def setUp(self):
        self.train_log = get_log(create_test_log(log_name=general_example_train_filename,
                                                 log_path=general_example_train_filepath))
        self.train_event_names = unique_events(self.train_log)
        self.train_add_col = get_additional_columns(self.train_log)

        self.test_log = get_log(create_test_log(log_name=general_example_test_filename,
                                                log_path=general_example_test_filepath_xes))
        self.test_event_names = unique_events(self.test_log)
        self.test_add_col = get_additional_columns(self.test_log)

    @staticmethod
    def _get_nn_default_config(encoding=ValueEncodings.SIMPLE_INDEX.value, binary: bool = False):
        config = dict()
        config['n_hidden_layers'] = 2
        config['n_hidden_units'] = 10
        config['activation'] = 'relu'
        config['n_epochs'] = 1
        config['encoding'] = encoding
        config['dropout_rate'] = 0.1
        config['is_binary_classifier'] = binary
        config['incremental_train'] = {'base_model': None}
        return config

    @staticmethod
    def _drop_columns_and_split(df: DataFrame) -> (DataFrame, ndarray):
        return df.drop(['trace_id', 'label'], 1), df['label']

    def test_nn_classifier_simple_index_binary_no_exceptions(self):
        encoding = create_test_encoding(value_encoding=ValueEncodings.SIMPLE_INDEX.value, prefix_length=2, padding=True)
        labelling = create_test_labelling(label_type=LabelTypes.DURATION.value,
                                          threshold_type=ThresholdTypes.THRESHOLD_MEAN.value)

        train_df = simple_index(self.train_log, labelling, encoding)
        test_df = simple_index(self.test_log, labelling, encoding)

        train_df, targets_df = self._drop_columns_and_split(train_df)
        targets_df = targets_df.values.ravel()

        test_df, _ = self._drop_columns_and_split(test_df)

        config = self._get_nn_default_config(binary=False)
        nn_classifier = NNClassifier(**config)

        # with HidePrints():
        nn_classifier.fit(train_df, targets_df)
        nn_classifier.predict(test_df)
        nn_classifier.predict_proba(test_df)

    def test_nn_classifier_simple_index_multiclass_no_exceptions(self):
        encoding = create_test_encoding(value_encoding=ValueEncodings.SIMPLE_INDEX.value, prefix_length=2, padding=True)
        labelling = create_test_labelling(label_type=LabelTypes.NEXT_ACTIVITY.value)

        train_df = simple_index(self.train_log, labelling, encoding)
        test_df = simple_index(self.test_log, labelling, encoding)

        train_df, targets_df = self._drop_columns_and_split(train_df)
        targets_df = targets_df.values.ravel()

        test_df, _ = self._drop_columns_and_split(test_df)

        config = self._get_nn_default_config(binary=False)
        nn_classifier = NNClassifier(**config)

        with HidePrints():
            nn_classifier.fit(train_df, targets_df)
            nn_classifier.predict(test_df)
            nn_classifier.predict_proba(test_df)
