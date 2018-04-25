from unittest import TestCase

from core.constants import SIMPLE_INDEX, CLASSIFICATION, COMPLEX
from encoders.common import encode_label_log, BOOLEAN
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
        self.assertListEqual(trace_5, ['5', 1, 2, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, True])

    def test_label_remaining_time_with_elapsed_time_custom_threshold(self):
        label = LabelContainer(add_elapsed_time=True, add_remaining_time=True, threshold_type=THRESHOLD_CUSTOM,
                               threshold=40000)

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 5))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'elapsed_time', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 90840.0, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, 75840.0, False])

    def test_remaining_time_zero_padding(self):
        label = LabelContainer(type=REMAINING_TIME, add_elapsed_time=True)

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=10, zero_padding=True)
        self.assertEqual(df.shape, (2, 13))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 3, 4, 5, 3, 2, 4, 5, 2, 1296240.0, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, 7, 4, 6, 0, 0, 0, 0, 0, 520920.0, True])

    def test_next_activity(self):
        label = LabelContainer(type=NEXT_ACTIVITY)

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 3])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, 7])

    def test_next_activity_zero_padding_elapsed_time(self):
        label = LabelContainer(type=NEXT_ACTIVITY, add_elapsed_time=True)

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=10, zero_padding=True)
        self.assertEqual(df.shape, (2, 13))
        self.assertTrue('elapsed_time' in df.columns.values.tolist())
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 3, 4, 5, 3, 2, 4, 5, 2, 1296240.0, 3])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, 7, 4, 6, 0, 0, 0, 0, 0, 520920.0, 0])

    def test_attribute_string(self):
        label = LabelContainer(type=ATTRIBUTE_STRING, attribute_name='creator')

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, "Fluxicon Nitro"])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, "Fluxicon Nitro"])

    def test_attribute_number(self):
        label = LabelContainer(type=ATTRIBUTE_NUMBER, attribute_name='number_value')

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, True])

    def test_duration(self):
        label = LabelContainer(type=DURATION)

        df = encode_label_log(self.log, SIMPLE_INDEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 3, True])


class TestLabelComplex(TestCase):
    """Cant be bothered to write better tests"""

    def setUp(self):
        self.log = get_logs("log_cache/general_example_test.xes")[0]
        self.event_names = unique_events(self.log)

    def test_no_label(self):
        label = LabelContainer(type=NO_LABEL)

        df = encode_label_log(self.log, COMPLEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual((2, 11), df.shape)

    def test_no_label_zero_padding(self):
        # add things have no effect
        label = LabelContainer(type=NO_LABEL, add_elapsed_time=True, add_remaining_time=True)

        df = encode_label_log(self.log, COMPLEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=10, zero_padding=True)
        self.assertEqual(df.shape, (2, 51))

    def test_remaining_time(self):
        label = LabelContainer()

        df = encode_label_log(self.log, COMPLEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 12))

    def test_label_remaining_time_with_elapsed_time_custom_threshold(self):
        label = LabelContainer(add_elapsed_time=True, add_remaining_time=True, threshold_type=THRESHOLD_CUSTOM,
                               threshold=40000)

        df = encode_label_log(self.log, COMPLEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 13))

    def test_remaining_time_zero_padding(self):
        label = LabelContainer(type=REMAINING_TIME, add_elapsed_time=True)

        df = encode_label_log(self.log, COMPLEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=10, zero_padding=True)
        self.assertEqual(df.shape, (2, 53))

    def test_next_activity(self):
        label = LabelContainer(type=NEXT_ACTIVITY)

        df = encode_label_log(self.log, COMPLEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 12))

    def test_next_activity_zero_padding_elapsed_time(self):
        label = LabelContainer(type=NEXT_ACTIVITY, add_elapsed_time=True)

        df = encode_label_log(self.log, COMPLEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=10, zero_padding=True)
        self.assertEqual(df.shape, (2, 53))
        self.assertTrue('elapsed_time' in df.columns.values.tolist())

    def test_attribute_string(self):
        label = LabelContainer(type=ATTRIBUTE_STRING, attribute_name='creator')

        df = encode_label_log(self.log, COMPLEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 12))

    def test_attribute_number(self):
        label = LabelContainer(type=ATTRIBUTE_NUMBER, attribute_name='number_value')

        df = encode_label_log(self.log, COMPLEX, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 12))


class TestLabelBoolean(TestCase):
    def setUp(self):
        self.log = get_logs("log_cache/general_example_test.xes")[0]
        self.event_names = unique_events(self.log)

    def test_no_label(self):
        label = LabelContainer(type=NO_LABEL)

        df = encode_label_log(self.log, BOOLEAN, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 8))

    def test_remaining_time(self):
        label = LabelContainer()

        df = encode_label_log(self.log, BOOLEAN, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 9))

    def test_label_remaining_time_with_elapsed_time_custom_threshold(self):
        label = LabelContainer(add_elapsed_time=True, add_remaining_time=True, threshold_type=THRESHOLD_CUSTOM,
                               threshold=40000)

        df = encode_label_log(self.log, BOOLEAN, CLASSIFICATION, label, event_names=self.event_names, prefix_length=4)
        self.assertEqual(df.shape, (2, 10))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, True, True, False, False, False, 361560.0, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, True, False, False, True, 248400.0, False])

    def test_next_activity(self):
        label = LabelContainer(type=NEXT_ACTIVITY)

        df = encode_label_log(self.log, BOOLEAN, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 9))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, False, False, False, False, False, False, 2])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, False, False, False, False, False, 3])

    def test_next_activity_zero_padding_elapsed_time(self):
        label = LabelContainer(type=NEXT_ACTIVITY, add_elapsed_time=True)

        df = encode_label_log(self.log, BOOLEAN, CLASSIFICATION, label, event_names=self.event_names, prefix_length=3)
        self.assertEqual(df.shape, (2, 10))
        self.assertTrue('elapsed_time' in df.columns.values.tolist())
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, True, False, False, False, False, 181200.0, 4])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, False, False, False, True, 171660.0, 4])

    def test_attribute_string(self):
        label = LabelContainer(type=ATTRIBUTE_STRING, attribute_name='creator')

        df = encode_label_log(self.log, BOOLEAN, CLASSIFICATION, label, event_names=self.event_names, prefix_length=3)
        self.assertEqual(df.shape, (2, 9))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, True, False, False, False, False, 'Fluxicon Nitro'])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, False, False, False, True, 'Fluxicon Nitro'])

    def test_attribute_number(self):
        label = LabelContainer(type=ATTRIBUTE_NUMBER, attribute_name='number_value')

        df = encode_label_log(self.log, BOOLEAN, CLASSIFICATION, label, event_names=self.event_names,
                              prefix_length=2)
        self.assertEqual(df.shape, (2, 9))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, False, False, False, False, False, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, False, False, False, False, True])
