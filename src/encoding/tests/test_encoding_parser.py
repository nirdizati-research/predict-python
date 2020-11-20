from django.test import TestCase
from pandas import DataFrame

from src.encoding.boolean_frequency import boolean, frequency
from src.encoding.complex_last_payload import complex
from src.encoding.encoding_parser import EncodingParser
from src.encoding.models import ValueEncodings
from src.encoding.simple_index import simple_index
from src.logs.log_service import get_log
from src.predictive_model.models import PredictiveModels
from src.utils.event_attributes import unique_events, get_additional_columns
from src.utils.tests_utils import general_example_train_filepath, create_test_log, general_example_train_filename, \
    create_test_labelling, create_test_encoding, general_example_test_filename, general_example_test_filepath_xes


class TestEncodingParser(TestCase):
    @staticmethod
    def _get_parser(encoding=ValueEncodings.COMPLEX.value, binary_target=False,
                    task=PredictiveModels.TIME_SERIES_PREDICTION.value):
        return EncodingParser(encoding, binary_target, task)

    @staticmethod
    def _drop_columns_and_split(df: DataFrame) -> (DataFrame, DataFrame):
        return df.drop(['trace_id', 'label'], 1), DataFrame(df['label'])

    def setUp(self):
        self.train_log = get_log(create_test_log(log_name=general_example_train_filename,
                                                 log_path=general_example_train_filepath))
        self.train_event_names = unique_events(self.train_log)
        self.labelling = create_test_labelling()
        self.train_add_col = get_additional_columns(self.train_log)

        self.test_log = get_log(create_test_log(log_name=general_example_test_filename,
                                                log_path=general_example_test_filepath_xes))
        self.test_event_names = unique_events(self.test_log)
        self.test_add_col = get_additional_columns(self.test_log)

    def test_simple_index_encoding_parsing(self):
        encoding = create_test_encoding(value_encoding=ValueEncodings.SIMPLE_INDEX.value, prefix_length=3, padding=True)

        parser = self._get_parser(encoding=ValueEncodings.SIMPLE_INDEX.value)
        train_df = simple_index(self.train_log, self.labelling, encoding)
        test_df = simple_index(self.test_log, self.labelling, encoding)

        train_df, targets_df = self._drop_columns_and_split(train_df)
        test_df, _ = self._drop_columns_and_split(test_df)
        test_df.iloc[0, 0] = 'test123'

        parser.parse_training_dataset(train_df)
        parser.parse_targets(targets_df)
        parser.parse_testing_dataset(test_df)

    def test_boolean_encoding_parsing(self):
        encoding = create_test_encoding(value_encoding=ValueEncodings.BOOLEAN.value, prefix_length=2, padding=True)

        parser = self._get_parser(encoding=ValueEncodings.BOOLEAN.value)
        train_df = boolean(self.train_log, self.train_event_names, self.labelling, encoding)
        test_df = boolean(self.test_log, self.test_event_names, self.labelling, encoding)

        train_df, targets_df = self._drop_columns_and_split(train_df)

        test_df, _ = self._drop_columns_and_split(test_df)

        parser.parse_training_dataset(train_df)
        parser.parse_targets(targets_df)
        parser.parse_testing_dataset(test_df)

    def test_frequency_encoding_parsing(self):
        encoding = create_test_encoding(value_encoding=ValueEncodings.FREQUENCY.value, prefix_length=2, padding=True)

        parser = self._get_parser(encoding=ValueEncodings.FREQUENCY.value)
        train_df = frequency(self.train_log, self.train_event_names, self.labelling, encoding)
        test_df = frequency(self.test_log, self.test_event_names, self.labelling, encoding)

        train_df, targets_df = self._drop_columns_and_split(train_df)

        test_df, _ = self._drop_columns_and_split(test_df)

        parser.parse_training_dataset(train_df)
        parser.parse_targets(targets_df)
        parser.parse_testing_dataset(test_df)

    def test_complex_encoding_parsing(self):
        encoding = create_test_encoding(value_encoding=ValueEncodings.COMPLEX.value, prefix_length=2, padding=True)

        parser = self._get_parser(encoding=ValueEncodings.COMPLEX.value)
        train_df = complex(self.train_log, self.labelling, encoding, self.train_add_col)
        test_df = complex(self.test_log, self.labelling, encoding, self.train_add_col)

        train_df, targets_df = self._drop_columns_and_split(train_df)

        test_df, _ = self._drop_columns_and_split(test_df)

        parser.parse_training_dataset(train_df)
        parser.parse_targets(targets_df)
        parser.parse_testing_dataset(test_df)
