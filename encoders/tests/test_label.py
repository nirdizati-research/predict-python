import unittest

from django.test import TestCase

from core.constants import CLASSIFICATION
from encoders.common import encode_label_log, BOOLEAN
from encoders.encoding_container import EncodingContainer, ZERO_PADDING, COMPLEX
from encoders.label_container import *
from utils.event_attributes import unique_events, get_additional_columns
from utils.file_service import get_log
from utils.tests_utils import general_example_test_filepath


# TODO: refactor tests


@unittest.skip('needs refactoring')
class TestLabelSimpleIndex(TestCase):
    def setUp(self):
        self.log = get_log(general_example_test_filepath)
        self.event_names = unique_events(self.log)
        self.encoding = EncodingContainer(prefix_length=2)

    def test_no_label(self):
        label = LabelContainer(type=NO_LABEL)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 3))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 52903968, 34856381, ])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 52903968, 32171502])

    def test_no_label_zero_padding(self):
        # add things have no effect
        label = LabelContainer(type=NO_LABEL, add_elapsed_time=True, add_remaining_time=True)
        encoding = EncodingContainer(prefix_length=10, padding=ZERO_PADDING)

        df = encode_label_log(self.log, encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 11))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5,
                             ['5', 52903968, 34856381, 32171502, 1149821, 70355923, 32171502, 34856381, 1149821,
                              70355923, 34856381])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4,
                             ['4', 52903968, 32171502, 17803069, 1149821, 72523760, 2595305, 2595305, 2595305, 2595305,
                              2595305])

    def test_remaining_time(self):
        label = LabelContainer()
        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 52903968, 34856381, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 52903968, 32171502, True])

    def test_label_remaining_time_with_elapsed_time_custom_threshold(self):
        label = LabelContainer(add_elapsed_time=True, add_remaining_time=True, threshold_type=THRESHOLD_CUSTOM,
                               threshold=40000)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 5))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'elapsed_time', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 52903968, 34856381, 90840.0, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 52903968, 32171502, 75840.0, False])

    def test_remaining_time_zero_padding(self):
        label = LabelContainer(type=REMAINING_TIME, add_elapsed_time=True)
        encoding = EncodingContainer(prefix_length=10, padding=ZERO_PADDING)

        df = encode_label_log(self.log, encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 13))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5,
                             ['5', 52903968, 34856381, 32171502, 1149821, 70355923, 32171502, 34856381, 1149821,
                              70355923, 34856381, 1296240.0, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4,
                             ['4', 52903968, 32171502, 17803069, 1149821, 72523760, 2595305, 2595305, 2595305, 2595305,
                              2595305, 520920.0, True])

    def test_next_activity(self):
        label = LabelContainer(type=NEXT_ACTIVITY)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 52903968, 34856381, 32171502])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 52903968, 32171502, 17803069])

    def test_next_activity_zero_padding_elapsed_time(self):
        label = LabelContainer(type=NEXT_ACTIVITY, add_elapsed_time=True)
        encoding = EncodingContainer(prefix_length=10, padding=ZERO_PADDING)

        df = encode_label_log(self.log, encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 13))
        self.assertTrue('elapsed_time' in df.columns.values.tolist())
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5,
                             ['5', 52903968, 34856381, 32171502, 1149821, 70355923, 32171502, 34856381, 1149821,
                              70355923, 34856381, 1296240.0, 32171502])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4,
                             ['4', 52903968, 32171502, 17803069, 1149821, 72523760, 2595305, 2595305, 2595305, 2595305,
                              2595305, 520920.0, 2595305])

    def test_attribute_string(self):
        label = LabelContainer(type=ATTRIBUTE_STRING, attribute_name='creator')

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 52903968, 34856381, 73510641])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 52903968, 32171502, 73510641])

    def test_attribute_number(self):
        label = LabelContainer(type=ATTRIBUTE_NUMBER, attribute_name='AMOUNT')

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 52903968, 34856381, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 52903968, 32171502, True])

    def test_duration(self):
        label = LabelContainer(type=DURATION)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 52903968, 34856381, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 52903968, 32171502, True])

    def test_add_executed_events(self):
        label = LabelContainer(add_executed_events=True)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 5))
        self.assertTrue('executed_events' in df.columns.values.tolist())
        self.assertListEqual(df['executed_events'].tolist(), [2, 2])

    def test_add_resources_used(self):
        label = LabelContainer(add_resources_used=True)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 5))
        self.assertTrue('resources_used' in df.columns.values.tolist())
        self.assertListEqual(df['resources_used'].tolist(), [1, 1])

    def test_add_new_traces(self):
        label = LabelContainer(add_new_traces=True)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 5))
        self.assertTrue('new_traces' in df.columns.values.tolist())
        self.assertListEqual(df['new_traces'].tolist(), [0, 0])


class TestLabelComplex(TestCase):
    def setUp(self):
        self.log = get_log(general_example_test_filepath)
        self.event_names = unique_events(self.log)
        self.add_col = get_additional_columns(self.log)
        self.encoding = EncodingContainer(COMPLEX, prefix_length=2)
        self.encodingPadding = EncodingContainer(COMPLEX, prefix_length=10, padding=ZERO_PADDING)

    def test_no_label(self):
        label = LabelContainer(type=NO_LABEL)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual((2, 13), df.shape)

    def test_no_label_zero_padding(self):
        # add things have no effect
        label = LabelContainer(type=NO_LABEL, add_elapsed_time=True, add_remaining_time=True)

        df = encode_label_log(self.log, self.encodingPadding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual(df.shape, (2, 53))

    def test_remaining_time(self):
        label = LabelContainer()

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual(df.shape, (2, 14))

    def test_label_remaining_time_with_elapsed_time_custom_threshold(self):
        label = LabelContainer(add_elapsed_time=True, add_remaining_time=True, threshold_type=THRESHOLD_CUSTOM,
                               threshold=40000)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual(df.shape, (2, 15))

    def test_remaining_time_zero_padding(self):
        label = LabelContainer(type=REMAINING_TIME, add_elapsed_time=True)

        df = encode_label_log(self.log, self.encodingPadding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual(df.shape, (2, 55))

    def test_add_executed_events(self):
        label = LabelContainer(add_executed_events=True)
        encoding = EncodingContainer(COMPLEX, prefix_length=2, padding=ZERO_PADDING)

        df = encode_label_log(self.log, encoding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual(df.shape, (2, 15))
        self.assertTrue('executed_events' in df.columns.values.tolist())
        self.assertListEqual(df['executed_events'].tolist(), [2, 2])

    def test_add_resources_used(self):
        label = LabelContainer(add_resources_used=True)
        encoding = EncodingContainer(COMPLEX, prefix_length=2, padding=ZERO_PADDING)

        df = encode_label_log(self.log, encoding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual(df.shape, (2, 15))
        self.assertTrue('resources_used' in df.columns.values.tolist())
        self.assertListEqual(df['resources_used'].tolist(), [1, 1])

    def test_add_new_traces(self):
        label = LabelContainer(add_new_traces=True)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual(df.shape, (2, 15))
        self.assertTrue('new_traces' in df.columns.values.tolist())
        self.assertListEqual(df['new_traces'].tolist(), [0, 0])

    def test_next_activity(self):
        label = LabelContainer(type=NEXT_ACTIVITY)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual(df.shape, (2, 14))

    def test_next_activity_zero_padding_elapsed_time(self):
        label = LabelContainer(type=NEXT_ACTIVITY, add_elapsed_time=True)

        df = encode_label_log(self.log, self.encodingPadding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual(df.shape, (2, 55))
        self.assertTrue('elapsed_time' in df.columns.values.tolist())

    def test_attribute_string(self):
        label = LabelContainer(type=ATTRIBUTE_STRING, attribute_name='creator')

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual(df.shape, (2, 14))

    def test_attribute_number(self):
        label = LabelContainer(type=ATTRIBUTE_NUMBER, attribute_name='AMOUNT')

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names,
                              additional_columns=self.add_col)
        self.assertEqual(df.shape, (2, 14))


class TestLabelBoolean(TestCase):
    def setUp(self):
        self.log = get_log(general_example_test_filepath)
        self.event_names = unique_events(self.log)
        self.encoding = EncodingContainer(BOOLEAN, prefix_length=2)

    def test_no_label(self):
        label = LabelContainer(type=NO_LABEL)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 8))

    def test_remaining_time(self):
        label = LabelContainer()

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 9))

    def test_label_remaining_time_with_elapsed_time_custom_threshold(self):
        label = LabelContainer(add_elapsed_time=True, add_remaining_time=True, threshold_type=THRESHOLD_CUSTOM,
                               threshold=40000)
        encoding = EncodingContainer(BOOLEAN, prefix_length=4)

        df = encode_label_log(self.log, encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 10))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, True, True, False, False, False, 361560.0, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, True, False, False, True, 248400.0, False])

    def test_next_activity(self):
        label = LabelContainer(type=NEXT_ACTIVITY)

        df = encode_label_log(self.log, EncodingContainer(BOOLEAN), CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 9))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, False, False, False, False, False, False, 34856381])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, False, False, False, False, False, 32171502])

    @unittest.skip('needs refactoring')
    def test_next_activity_zero_padding_elapsed_time(self):
        label = LabelContainer(type=NEXT_ACTIVITY, add_elapsed_time=True)
        encoding = EncodingContainer(BOOLEAN, prefix_length=3)

        df = encode_label_log(self.log, encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 10))
        self.assertTrue('elapsed_time' in df.columns.values.tolist())
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, True, False, False, False, False, 181200.0, 1149821])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, False, False, False, True, 171660.0, 1149821])

    @unittest.skip('needs refactoring')
    def test_attribute_string(self):
        label = LabelContainer(type=ATTRIBUTE_STRING, attribute_name='creator')
        encoding = EncodingContainer(BOOLEAN, prefix_length=3)

        df = encode_label_log(self.log, encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 9))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, True, False, False, False, False, 73510641])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, False, False, False, True, 73510641])

    def test_attribute_number(self):
        label = LabelContainer(type=ATTRIBUTE_NUMBER, attribute_name='AMOUNT')

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 9))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, False, False, False, False, False, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, False, False, False, False, True])

    def test_add_executed_events(self):
        label = LabelContainer(add_executed_events=True)
        encoding = EncodingContainer(BOOLEAN, prefix_length=3)

        df = encode_label_log(self.log, encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 10))
        self.assertTrue('executed_events' in df.columns.values.tolist())
        self.assertListEqual(df['executed_events'].tolist(), [2, 2])

    def test_add_resources_used(self):
        label = LabelContainer(add_resources_used=True)
        encoding = EncodingContainer(BOOLEAN, prefix_length=3)

        df = encode_label_log(self.log, encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 10))
        self.assertTrue('resources_used' in df.columns.values.tolist())
        self.assertListEqual(df['resources_used'].tolist(), [2, 2])

    def test_add_new_traces(self):
        label = LabelContainer(add_new_traces=True)

        df = encode_label_log(self.log, self.encoding, CLASSIFICATION, label, event_names=self.event_names)
        self.assertEqual(df.shape, (2, 10))
        self.assertTrue('new_traces' in df.columns.values.tolist())
        self.assertListEqual(df['new_traces'].tolist(), [0, 0])
        self.assertFalse(df.isnull().values.any())
