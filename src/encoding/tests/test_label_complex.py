from django.test import TestCase

from src.encoding.common import encode_label_logs
from src.encoding.models import ValueEncodings, TaskGenerationTypes
from src.labelling.label_container import *
from src.predictive_model.models import PredictiveModels
from src.utils.event_attributes import get_additional_columns
from src.utils.file_service import get_log
from src.utils.tests_utils import general_example_test_filepath, create_test_log, general_example_test_filename, \
    create_test_encoding, create_test_labelling, general_example_train_filename, general_example_train_filepath, \
    create_test_job, create_test_predictive_model


class TestLabelComplex(TestCase):
    def setUp(self):
        self.train_log = get_log(create_test_log(log_name=general_example_train_filename,
                                                 log_path=general_example_train_filepath))
        self.test_log = get_log(create_test_log(log_name=general_example_test_filename,
                                                log_path=general_example_test_filepath))
        self.add_col = get_additional_columns(self.train_log)
        self.encoding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2)
        self.encodingPadding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=10,
            padding=True)

    def test_no_label(self):
        labelling = create_test_labelling(label_type=LabelTypes.NO_LABEL.value)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=self.encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual((2, 13), df.shape)

    def test_no_label_zero_padding(self):
        # add things have no effect
        labelling = create_test_labelling(label_type=LabelTypes.NO_LABEL.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            add_elapsed_time=True,
            add_remaining_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=10,
            padding=True)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 53))

    def test_remaining_time(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=self.encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 14))

    def test_label_remaining_time_with_elapsed_time_custom_threshold(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value,
                                          threshold_type=ThresholdTypes.THRESHOLD_CUSTOM.value,
                                          threshold=40000)
        encoding = create_test_encoding(value_encoding=ValueEncodings.COMPLEX.value,
                                        add_elapsed_time=True,
                                        add_remaining_time=True,
                                        task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
                                        prefix_length=10)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 15))

    def test_remaining_time_zero_padding(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=self.encodingPadding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 55))

    def test_add_executed_events(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        encoding = create_test_encoding(value_encoding=ValueEncodings.COMPLEX.value,
                                        prefix_length=2,
                                        padding=True,
                                        add_executed_events=True,
                                        add_elapsed_time=True)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 15))
        self.assertTrue('executed_events' in df.columns.values.tolist())
        self.assertListEqual(df['executed_events'].tolist(), [2, 2])

    def test_add_resources_used(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        encoding = create_test_encoding(value_encoding=ValueEncodings.COMPLEX.value,
                                        prefix_length=2,
                                        padding=True,
                                        add_elapsed_time=True,
                                        add_resources_used=True)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 15))
        self.assertTrue('resources_used' in df.columns.values.tolist())
        self.assertListEqual(df['resources_used'].tolist(), [1, 1])

    def test_add_new_traces(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        encoding = create_test_encoding(value_encoding=ValueEncodings.COMPLEX.value,
                                        prefix_length=2,
                                        add_new_traces=True,
                                        add_elapsed_time=True)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 15))
        self.assertTrue('new_traces' in df.columns.values.tolist())
        self.assertListEqual(df['new_traces'].tolist(), [0, 0])

    def test_next_activity(self):
        labelling = create_test_labelling(label_type=LabelTypes.NEXT_ACTIVITY.value)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=self.encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 14))

    def test_next_activity_zero_padding_elapsed_time(self):
        labelling = create_test_labelling(label_type=LabelTypes.NEXT_ACTIVITY.value)
        encoding = create_test_encoding(value_encoding=ValueEncodings.COMPLEX.value,
                                        add_elapsed_time=True,
                                        add_remaining_time=True,
                                        task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
                                        prefix_length=10,
                                        padding=True)

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 55))
        self.assertTrue('elapsed_time' in df.columns.values.tolist())

    def test_attribute_string(self):
        labelling = create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value, attribute_name='creator')

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=self.encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 14))

    def test_attribute_number(self):
        labelling = create_test_labelling(label_type=LabelTypes.ATTRIBUTE_NUMBER.value, attribute_name='AMOUNT')

        _, df = encode_label_logs(self.train_log, self.test_log, create_test_job(
            encoding=self.encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 14))
