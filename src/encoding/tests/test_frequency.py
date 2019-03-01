from django.test import TestCase

from src.encoding.boolean_frequency import frequency
from src.encoding.common import encode_label_logs, LabelContainer, LabelTypes
from src.encoding.encoding_container import EncodingContainer, ZERO_PADDING
from src.encoding.models import ValueEncodings
from src.jobs.models import JobTypes
from src.utils.event_attributes import unique_events
from src.utils.file_service import get_log
from src.utils.tests_utils import general_example_test_filepath, general_example_train_filepath


class TestFrequencySplit(TestCase):
    def setUp(self):
        test_log = get_log(general_example_test_filepath)
        training_log = get_log(general_example_train_filepath)
        self.training_df, self.test_df = encode_label_logs(training_log, test_log,
                                                           EncodingContainer(ValueEncodings.FREQUENCY.value),
                                                           JobTypes.PREDICTION.value,
                                                           LabelContainer(add_elapsed_time=True))

    def test_shape(self):
        self.assert_shape(self.training_df, (4, 11))
        self.assert_shape(self.test_df, (2, 11))

    def assert_shape(self, df, shape: tuple):
        names = ['register request', 'examine casually', 'check ticket', 'decide',
                 'reinitiate request', 'examine thoroughly',
                 'reject request', 'trace_id', 'label', 'elapsed_time']
        for name in names:
            self.assertIn(name, df.columns.values.tolist())
        self.assertEqual(shape, df.shape)


class TestGeneralTest(TestCase):
    """Making sure it actually works"""

    def setUp(self):
        self.log = get_log(general_example_test_filepath)
        self.event_names = unique_events(self.log)
        self.label = LabelContainer(add_elapsed_time=True)
        self.encoding = EncodingContainer(ValueEncodings.FREQUENCY.value)

    def test_header(self):
        df = frequency(self.log, self.event_names, self.label, self.encoding)
        names = ['register request', 'examine casually', 'check ticket', 'decide',
                 'reinitiate request', 'examine thoroughly',
                 'reject request', 'trace_id', 'label', 'elapsed_time']
        for name in names:
            self.assertIn(name, df.columns.values.tolist())

    def test_prefix1(self):
        df = frequency(self.log, self.event_names, self.label, self.encoding)

        self.assertEqual(df.shape, (2, 10))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertTrue(row1['register request'])
        self.assertFalse(row1['examine casually'])
        self.assertEqual(1576440.0, row1.label)
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertTrue(row2['register request'])
        self.assertFalse(row2['examine casually'])
        self.assertEqual(520920.0, row2.label)

    def test_prefix1_no_label(self):
        df = frequency(self.log, self.event_names, LabelContainer(LabelTypes.NO_LABEL.value), self.encoding)

        self.assertEqual(df.shape, (2, 8))
        self.assertNotIn('label', df.columns.values.tolist())

    def test_prefix1_no_elapsed_time(self):
        label = LabelContainer()
        df = frequency(self.log, self.event_names, label, self.encoding)

        self.assertEqual(df.shape, (2, 9))
        self.assertNotIn('elapsed_time', df.columns.values.tolist())

    def test_prefix2(self):
        encoding = EncodingContainer(ValueEncodings.FREQUENCY.value, prefix_length=2)
        df = frequency(self.log, self.event_names, self.label, encoding)

        self.assertEqual(df.shape, (2, 10))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertTrue(row1['register request'])
        self.assertTrue(row1['examine casually'])
        self.assertEqual(1485600.0, row1.label)
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertTrue(row2['register request'])
        self.assertFalse(row2['examine casually'])
        self.assertTrue(row2['check ticket'])
        self.assertEqual(445080.0, row2.label)

    def test_prefix5(self):
        encoding = EncodingContainer(ValueEncodings.FREQUENCY.value, prefix_length=5)
        df = frequency(self.log, self.event_names, self.label, encoding)

        self.assertEqual(df.shape, (2, 10))
        row1 = df[df.trace_id == '5'].iloc[0]
        # 1 == True, 0 == False
        self.assertListEqual(['5', True, True, True, True, True, False, False, 458160.0, 1118280.0],
                             row1.values.tolist())

    def test_prefix10(self):
        encoding = EncodingContainer(ValueEncodings.FREQUENCY.value, prefix_length=10)
        df = frequency(self.log, self.event_names, self.label, encoding)

        self.assertEqual(df.shape, (1, 10))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 1, 3, 2, 2, 2, 0, 0, 1296240.0, 280200.0],
                             row1.values.tolist())
        self.assertFalse(df.isnull().values.any())

    def test_prefix10_padding(self):
        encoding = EncodingContainer(ValueEncodings.FREQUENCY.value, prefix_length=10, padding=ZERO_PADDING)
        df = frequency(self.log, self.event_names, self.label, encoding)

        self.assertEqual(df.shape, (2, 10))
        row1 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', True, False, True, True, False, True, True, 520920.0, 0.0], row1.values.tolist())
        self.assertFalse(df.isnull().values.any())
