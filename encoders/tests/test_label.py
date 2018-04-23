from unittest import TestCase

from core.constants import SIMPLE_INDEX, CLASSIFICATION
from encoders.common import encode_label_log
from encoders.label_container import *
from encoders.log_util import unique_events
from logs.file_service import get_logs


class TestLabelSimpleIndex(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example_test.xes")[0]
        self.event_names = unique_events(self.log)

    def test_no_label(self):
        label = LabelContainer(type=NO_LABEL)

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 3))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3])

    def test_no_label_zero_padding(self):
        # add things have no effect
        label = LabelContainer(type=NO_LABEL, add_elapsed_time=True, add_remaining_time=True)

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=10, zero_padding=True)
        self.assertEqual(df.shape, (2, 11))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 3, 4, 5, 3, 2, 4, 5, 2])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, 7, 4, 6, 0, 0, 0, 0, 0])

    def test_remaining_time(self):
        label = LabelContainer()

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 1485600.0])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, 445080.0])

    def test_label_remaining_time_with_elapsed_time(self):
        label = LabelContainer(add_elapsed_time=True, add_remaining_time=True)

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 5))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'elapsed_time', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 90840.0, 1485600.0])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, 75840.0, 445080.0])

    def test_remaining_time_zero_padding(self):
        label = LabelContainer(type=REMAINING_TIME, add_elapsed_time=True)

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=10, zero_padding=True)
        self.assertEqual(df.shape, (2, 12))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 3, 4, 5, 3, 2, 4, 5, 2, 280200.0])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, 7, 4, 6, 0, 0, 0, 0, 0, 0.0])


