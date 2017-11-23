from unittest import TestCase

from core.constants import SIMPLE_INDEX, CLASSIFICATION, NEXT_ACTIVITY
from encoders.common import encode_logs
from encoders.log_util import unique_events
from encoders.simple_index import simple_index
from logs.file_service import get_logs


class TestSimpleGeneralExample(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example.xes")[0]
        self.event_names = unique_events(self.log)

    def test_shape(self):
        df = simple_index(self.log, self.event_names)

        self.assertIn("trace_id", df.columns.values)
        self.assertIn("event_nr", df.columns.values)
        self.assertIn("remaining_time", df.columns.values)
        self.assertIn("elapsed_time", df.columns.values)
        self.assertIn("prefix_1", df.columns.values)
        self.assertEqual((6, 5), df.shape)

    def test_prefix_length(self):
        df = simple_index(self.log, self.event_names, prefix_length=3)
        self.assertIn("prefix_1", df.columns.values)
        self.assertIn("prefix_2", df.columns.values)
        self.assertIn("prefix_3", df.columns.values)
        self.assertEqual((6, 7), df.shape)

        row = df[(df.event_nr == 3) & (df.trace_id == '3')].iloc[0]
        self.assertEqual(1, row.prefix_1)
        self.assertEqual(2, row.prefix_2)
        self.assertEqual(3, row.prefix_3)
        self.assertEqual(782820.0, row.remaining_time)
        self.assertEqual(585960.0, row.elapsed_time)

    def test_row(self):
        df = simple_index(self.log, self.event_names)
        row = df[(df.event_nr == 1) & (df.trace_id == '3')].iloc[0]

        self.assertEqual(1.0, row.prefix_1)
        self.assertEqual(2040.0, row.elapsed_time)
        self.assertEqual(1366740.0, row.remaining_time)

    def test_encodes_next_activity(self):
        """Encodes for next activity"""
        df = simple_index(self.log, self.event_names, next_activity=True)

        self.assertEqual((6, 3), df.shape)
        self.assertNotIn("remaining_time", df.columns.values)
        self.assertNotIn("elapsed_time", df.columns.values)
        self.assertIn("label", df.columns.values)
        self.assertNotIn("prefix_1", df.columns.values)

        row = df[df.trace_id == '3'].iloc[0]
        self.assertEqual(1, row.label)
        self.assertEqual(1, row.event_nr)

    def test_encodes_next_activity_prefix(self):
        """Encodes for next activity with prefix length"""
        df = simple_index(self.log, self.event_names, prefix_length=6, next_activity=True)

        self.assertEqual((6, 8), df.shape)
        self.assertIn("prefix_1", df.columns.values)
        self.assertIn("prefix_2", df.columns.values)
        self.assertIn("prefix_3", df.columns.values)
        row = df[df.trace_id == '3'].iloc[0]
        self.assertListEqual(['3', 6, 1, 2, 3, 4, 5, 6], row.values.tolist())


class TestSplitLogExample(TestCase):
    def setUp(self):
        self.test_log = get_logs("log_cache/general_example_test.xes")[0]
        self.training_log = get_logs("log_cache/general_example_training.xes")[0]

    def test_shape_training(self):
        training_df, test_df = encode_logs(self.training_log, self.test_log, SIMPLE_INDEX, CLASSIFICATION)

        self.assert_shape(training_df, (4, 5))
        self.assert_shape(test_df, (2, 5))

    def assert_shape(self, dataframe, shape):
        self.assertIn("trace_id", dataframe.columns.values)
        self.assertIn("event_nr", dataframe.columns.values)
        self.assertIn("remaining_time", dataframe.columns.values)
        self.assertIn("elapsed_time", dataframe.columns.values)
        self.assertIn("prefix_1", dataframe.columns.values)
        self.assertEqual(shape, dataframe.shape)

    def test_prefix_length_training(self):
        training_df, test_df = encode_logs(self.training_log, self.test_log, SIMPLE_INDEX, CLASSIFICATION,
                                           prefix_length=3)
        self.assertIn("prefix_1", training_df.columns.values)
        self.assertIn("prefix_2", training_df.columns.values)
        self.assertIn("prefix_3", training_df.columns.values)
        self.assertEqual((4, 7), training_df.shape)
        self.assertEqual((2, 7), test_df.shape)

        row = training_df[(training_df.event_nr == 3) & (training_df.trace_id == '3')].iloc[0]
        self.assertEqual(1, row.prefix_1)
        self.assertEqual(2, row.prefix_2)
        self.assertEqual(3, row.prefix_3)
        self.assertEqual(782820.0, row.remaining_time)
        self.assertEqual(585960.0, row.elapsed_time)

    def test_row_test(self):
        training_df, test_df = encode_logs(self.training_log, self.test_log, SIMPLE_INDEX, CLASSIFICATION)

        row = test_df[(test_df.event_nr == 1) & (test_df.trace_id == '4')].iloc[0]

        self.assertEqual(1.0, row.prefix_1)
        self.assertEqual(75840.0, row.elapsed_time)
        self.assertEqual(445080.0, row.remaining_time)

    def test_encodes_next_activity(self):
        """Encodes for next activity with test set"""
        training_df, test_df = encode_logs(self.training_log, self.test_log, SIMPLE_INDEX, NEXT_ACTIVITY)

        self.assertEqual((2, 3), test_df.shape)
        self.assertNotIn("remaining_time", test_df.columns.values)
        self.assertNotIn("elapsed_time", test_df.columns.values)
        self.assertIn("label", test_df.columns.values)
        self.assertNotIn("prefix_1", test_df.columns.values)

        row = test_df[test_df.trace_id == '4'].iloc[0]
        self.assertEqual(1, row.label)
        self.assertEqual(1, row.event_nr)

    def test_encodes_next_activity_prefix(self):
        """Encodes for next activity with prefix length with training set"""
        training_df, test_df = encode_logs(self.training_log, self.test_log, SIMPLE_INDEX, NEXT_ACTIVITY,
                                           prefix_length=6)

        self.assertEqual((4, 8), training_df.shape)
        self.assertIn("prefix_1", training_df.columns.values)
        self.assertIn("prefix_2", training_df.columns.values)
        self.assertIn("prefix_3", training_df.columns.values)
        row = training_df[training_df.trace_id == '3'].iloc[0]
        self.assertListEqual(['3', 6, 1, 2, 3, 4, 5, 6], row.values.tolist())
