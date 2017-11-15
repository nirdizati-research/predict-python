from unittest import TestCase

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
