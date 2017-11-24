from unittest import TestCase

from core.constants import SIMPLE_INDEX, NEXT_ACTIVITY
from encoders.common import encode_logs
from encoders.log_util import unique_events
from encoders.simple_index import simple_index
from logs.file_service import get_logs


class TestSimpleGeneralExample(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example.xes")[0]
        self.event_names = unique_events(self.log)

    def test_encodes_next_activity(self):
        """Encodes for next activity"""
        df = simple_index(self.log, self.event_names, next_activity=True)

        self.assertEqual((6, 3), df.shape)
        self.assertNotIn("remaining_time", df.columns.values)
        self.assertNotIn("elapsed_time", df.columns.values)
        self.assertIn("label", df.columns.values)
        self.assertIn("prefix_1", df.columns.values)

        row = df[df.trace_id == '3'].iloc[0]
        self.assertEqual(2, row.label)
        self.assertEqual(1, row.prefix_1)

    def test_encodes_next_activity_prefix(self):
        """Encodes for next activity with prefix length"""
        df = simple_index(self.log, self.event_names, prefix_length=6, next_activity=True)

        self.assertEqual((6, 8), df.shape)
        self.assertIn("prefix_1", df.columns.values)
        self.assertIn("prefix_2", df.columns.values)
        self.assertIn("prefix_3", df.columns.values)
        row = df[df.trace_id == '3'].iloc[0]
        self.assertListEqual(['3', 1, 2, 3, 4, 5, 6, 3], row.values.tolist())


class TestSplitLogExample(TestCase):
    def setUp(self):
        self.test_log = get_logs("log_cache/general_example_test.xes")[0]
        self.training_log = get_logs("log_cache/general_example_training.xes")[0]

    def test_encodes_next_activity(self):
        """Encodes for next activity with test set"""
        training_df, test_df = encode_logs(self.training_log, self.test_log, SIMPLE_INDEX, NEXT_ACTIVITY)

        self.assertEqual((2, 3), test_df.shape)
        self.assertNotIn("remaining_time", test_df.columns.values)
        self.assertNotIn("elapsed_time", test_df.columns.values)
        self.assertIn("label", test_df.columns.values)
        self.assertIn("prefix_1", test_df.columns.values)

        row = test_df[test_df.trace_id == '4'].iloc[0]
        self.assertEqual(3, row.label)
        self.assertEqual(1, row.prefix_1)

    def test_encodes_next_activity_prefix(self):
        """Encodes for next activity with prefix length with training set"""
        training_df, test_df = encode_logs(self.training_log, self.test_log, SIMPLE_INDEX, NEXT_ACTIVITY,
                                           prefix_length=6)

        self.assertEqual((4, 8), training_df.shape)
        self.assertIn("prefix_1", training_df.columns.values)
        self.assertIn("prefix_2", training_df.columns.values)
        self.assertIn("prefix_3", training_df.columns.values)
        row = training_df[training_df.trace_id == '3'].iloc[0]
        self.assertListEqual(['3', 1, 2, 3, 4, 5, 6, 3], row.values.tolist())


class TestNextActivity(TestCase):
    """Making sure it actually works"""

    def setUp(self):
        self.log = get_logs("log_cache/general_example_test.xes")[0]
        self.event_names = unique_events(self.log)

    def test_header(self):
        df = simple_index(self.log, self.event_names, prefix_length=3, next_activity=True)

        self.assertEqual(df.shape, (2, 5))
        header = ['trace_id', 'prefix_1', 'prefix_2', 'prefix_3', 'label']
        self.assertListEqual(header, df.columns.values.tolist())

    def test_prefix1(self):
        df = simple_index(self.log, self.event_names, prefix_length=1, next_activity=True)

        self.assertEqual(df.shape, (2, 3))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 1, 2], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 1, 3], row2.values.tolist())

    def test_prefix0(self):
        df = simple_index(self.log, self.event_names, prefix_length=0, next_activity=True)

        self.assertEqual(df.shape, (2, 2))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 1], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 1], row2.values.tolist())

    def test_prefix2(self):
        df = simple_index(self.log, self.event_names, prefix_length=2, next_activity=True)

        self.assertEqual(df.shape, (2, 4))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 1, 2, 3], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 1, 3, 7], row2.values.tolist())

    def test_prefix5(self):
        df = simple_index(self.log, self.event_names, prefix_length=5, next_activity=True)

        self.assertEqual(df.shape, (2, 7))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 1, 2, 3, 4, 5, 3], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 1, 3, 7, 4, 6, 0], row2.values.tolist())

    def test_prefix10(self):
        df = simple_index(self.log, self.event_names, prefix_length=10, next_activity=True)

        self.assertEqual(df.shape, (2, 12))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 1, 2, 3, 4, 5, 3, 2, 4, 5, 2, 3], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 1, 3, 7, 4, 6, 0, 0, 0, 0, 0, 0], row2.values.tolist())
