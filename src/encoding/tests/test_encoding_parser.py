import unittest

from django.test import TestCase

from src.encoding.complex_last_payload import complex
from src.encoding.encoding_container import EncodingContainer
from src.encoding.encoding_parser import EncodingParser, Tasks
from src.encoding.models import ValueEncodings
from src.labelling.label_container import LabelContainer
from src.utils.event_attributes import unique_events, get_additional_columns
from src.utils.file_service import get_log
from src.utils.tests_utils import general_example_train_filepath, create_test_log, general_example_train_filename, \
    create_test_labelling, create_test_encoding


class TestEncodingParser(TestCase):
    @staticmethod
    def _get_parser(encoding=ValueEncodings.COMPLEX.value, binary_target=False, task=Tasks.CLASSIFICATION.value):
        return EncodingParser(encoding, binary_target, task)

    def setUp(self):
        self.log = get_log(create_test_log(log_name=general_example_train_filename,
                                           log_path=general_example_train_filepath))
        self.event_names = unique_events(self.log)
        self.label = create_test_labelling()
        self.add_col = get_additional_columns(self.log)
        self.encoding = create_test_encoding(value_encoding=ValueEncodings.COMPLEX.value)

    def test_simple_index(self):
        parser = self._get_parser(encoding=ValueEncodings.SIMPLE_INDEX.value)
        df = complex(self.log, self.label, self.encoding, self.add_col)
        parser.parse_training_dataset(df)
