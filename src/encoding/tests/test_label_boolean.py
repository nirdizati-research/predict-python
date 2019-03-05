from django.test import TestCase

from src.encoding.common import encode_label_logs
from src.encoding.models import ValueEncodings, TaskGenerationTypes
from src.labelling.label_container import *
from src.predictive_model.models import PredictiveModels
from src.utils.file_service import get_log
from src.utils.tests_utils import general_example_test_filepath, create_test_log, general_example_test_filename, \
    create_test_encoding, create_test_labelling, general_example_train_filename, general_example_train_filepath, \
    create_test_job, create_test_predictive_model


class TestLabelBoolean(TestCase):
    def setUp(self):
        self.train_log = get_log(create_test_log(log_name=general_example_train_filename,
                                                 log_path=general_example_train_filepath))
        self.test_log = get_log(create_test_log(log_name=general_example_test_filename,
                                                log_path=general_example_test_filepath))
        self.encoding = create_test_encoding(
            value_encoding=ValueEncodings.BOOLEAN.value,
            prefix_length=2,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value)

    def test_no_label(self):
        labelling = create_test_labelling(label_type=LabelTypes.NO_LABEL.value)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=self.encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 9))

    def test_remaining_time(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=self.encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 11))

    def test_label_remaining_time_with_elapsed_time_custom_threshold(self):
        labelling = create_test_labelling(
            label_type=LabelTypes.REMAINING_TIME.value,
            threshold_type=ThresholdTypes.THRESHOLD_CUSTOM.value,
            threshold=40000)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.BOOLEAN.value,
            prefix_length=3,
            add_elapsed_time=True,
            add_remaining_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value)

        _, df = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 10))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, True, True, False, False, False, 361560.0, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, True, False, False, True, 248400.0, False])

    def test_next_activity(self):
        labelling = create_test_labelling(label_type=LabelTypes.NEXT_ACTIVITY.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.BOOLEAN.value,
            prefix_length=3,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value)

        _, df = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 9))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, True, False, False, False, False, 3])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, False, False, False, True, 3])

    def test_next_activity_zero_padding_elapsed_time(self):
        labelling = create_test_labelling(label_type=LabelTypes.NEXT_ACTIVITY.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.BOOLEAN.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=3)

        _, df = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 10))
        self.assertTrue('elapsed_time' in df.columns.values.tolist())
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, True, False, False, False, False, 181200.0, 3])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, False, False, False, True, 171660.0, 3])

    def test_attribute_string(self):
        labelling = create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value, attribute_name='creator')
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.BOOLEAN.value,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=3)

        _, df = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 9))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, True, False, False, False, False, 'Fluxicon Nitro'])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, False, False, False, True, 'Fluxicon Nitro'])

    def test_attribute_number(self):
        labelling = create_test_labelling(label_type=LabelTypes.ATTRIBUTE_NUMBER.value, attribute_name='AMOUNT')

        _, df = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=self.encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 9))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', True, True, False, False, False, False, False, False])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', True, False, True, False, False, False, False, True])

    def test_add_executed_events(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.BOOLEAN.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2,
            add_executed_events=True)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 12))
        self.assertTrue('executed_events' in df.columns.values.tolist())
        self.assertListEqual(df['executed_events'].tolist(), [2, 2])

    def test_add_resources_used(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.BOOLEAN.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2,
            add_resources_used=True)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 12))
        self.assertTrue('resources_used' in df.columns.values.tolist())
        self.assertListEqual(df['resources_used'].tolist(), [2, 2])

    def test_add_new_traces(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.BOOLEAN.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2,
            add_new_traces=True)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 12))
        self.assertTrue('new_traces' in df.columns.values.tolist())
        self.assertListEqual(df['new_traces'].tolist(), [0, 0])
        self.assertFalse(df.isnull().values.any())
