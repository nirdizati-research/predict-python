from unittest import TestCase

from core.constants import SIMPLE_INDEX, CLASSIFICATION
from encoders.common import LabelContainer, encode_label_logs, NO_LABEL
from encoders.simple_index import simple_index
from logs.file_service import get_logs


class TestSplitLogExample(TestCase):
    def setUp(self):
        self.test_log = get_logs("log_cache/general_example_test.xes")[0]
        self.training_log = get_logs("log_cache/general_example_training.xes")[0]
        self.label = LabelContainer(add_elapsed_time=True)

    def test_shape_training(self):
        training_df, test_df = encode_label_logs(self.training_log, self.test_log, SIMPLE_INDEX,
                                                 CLASSIFICATION, self.label, prefix_length=1)
        self.assert_shape(training_df, (4, 4))
        self.assert_shape(test_df, (2, 4))

    def assert_shape(self, dataframe, shape):
        self.assertIn("trace_id", dataframe.columns.values)
        self.assertIn("label", dataframe.columns.values)
        self.assertIn("elapsed_time", dataframe.columns.values)
        self.assertIn("prefix_1", dataframe.columns.values)
        self.assertEqual(shape, dataframe.shape)

    def test_prefix_length_training(self):
        training_df, test_df = encode_label_logs(self.training_log, self.test_log, SIMPLE_INDEX,
                                                 CLASSIFICATION, self.label, prefix_length=3)
        self.assertIn("prefix_1", training_df.columns.values)
        self.assertIn("prefix_2", training_df.columns.values)
        self.assertIn("prefix_3", training_df.columns.values)
        self.assertEqual((4, 6), training_df.shape)
        self.assertEqual((2, 6), test_df.shape)

        row = training_df[(training_df.trace_id == '3')].iloc[0]
        self.assertEqual(52903968, row.prefix_1)
        self.assertEqual(34856381, row.prefix_2)
        self.assertEqual(32171502, row.prefix_3)
        self.assertEqual(False, row.label)
        self.assertEqual(7320.0, row.elapsed_time)

    def test_row_test(self):
        training_df, test_df = encode_label_logs(self.training_log, self.test_log, SIMPLE_INDEX,
                                                 CLASSIFICATION, self.label, prefix_length=1)
        row = test_df[(test_df.trace_id == '4')].iloc[0]

        self.assertEqual(52903968, row.prefix_1)
        self.assertEqual(0.0, row.elapsed_time)
        self.assertEqual(True, row.label)


class TestGeneralTest(TestCase):
    """Making sure it actually works"""

    def setUp(self):
        self.log = get_logs("log_cache/general_example_test.xes")[0]
        self.label = LabelContainer(add_elapsed_time=True)

    def test_header(self):
        df = simple_index(self.log, self.label)

        self.assertIn("trace_id", df.columns.values)
        self.assertIn("label", df.columns.values)
        self.assertIn("elapsed_time", df.columns.values)
        self.assertIn("prefix_1", df.columns.values)

    def test_prefix1(self):
        df = simple_index(self.log, self.label, prefix_length=1)

        self.assertEqual(df.shape, (2, 4))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 'register request', 0.0, 1576440.0], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 'register request', 0.0, 520920.0], row2.values.tolist())

    def test_prefix1_no_label(self):
        df = simple_index(self.log, LabelContainer(NO_LABEL), prefix_length=1)

        self.assertEqual(df.shape, (2, 2))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 'register request'], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 'register request'], row2.values.tolist())

    def test_prefix1_no_elapsed_time(self):
        label = LabelContainer()
        df = simple_index(self.log, label, prefix_length=1)

        self.assertEqual(df.shape, (2, 3))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 'register request', 1576440.0], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 'register request', 520920.0], row2.values.tolist())

    def test_prefix0(self):
        self.assertRaises(ValueError,
                          simple_index, self.log, self.label, prefix_length=0)

    def test_prefix2(self):
        df = simple_index(self.log, self.label, prefix_length=2)

        self.assertEqual(df.shape, (2, 5))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 'register request', 'examine casually', 90840.0, 1485600.0], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 'register request', 'check ticket', 75840.0, 445080.0], row2.values.tolist())

    def test_prefix5(self):
        df = simple_index(self.log, self.label, prefix_length=5)

        self.assertEqual(df.shape, (2, 8))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(
            ['5', 'register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request', 458160.0,
             1118280.0], row1.values.tolist())

    def test_prefix10(self):
        df = simple_index(self.log, self.label, prefix_length=10)

        self.assertEqual(df.shape, (1, 13))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(
            ['5', 'register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request',
             'check ticket', 'examine casually', 'decide', 'reinitiate request', 'examine casually', 1296240.0,
             280200.0], row1.values.tolist())

    def test_prefix10_padding(self):
        df = simple_index(self.log, self.label, prefix_length=10, zero_padding=True)

        self.assertEqual(df.shape, (2, 13))
        row1 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(
            ['4', 'register request', 'check ticket', 'examine thoroughly', 'decide', 'reject request', '0', '0', '0',
             '0', '0', 520920.0, 0.0], row1.values.tolist())

