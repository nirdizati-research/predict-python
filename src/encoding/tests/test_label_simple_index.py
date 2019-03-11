from django.test import TestCase

from src.encoding.common import encode_label_logs
from src.encoding.models import ValueEncodings, TaskGenerationTypes
from src.labelling.label_container import *
from src.predictive_model.models import PredictiveModels
from src.utils.file_service import get_log
from src.utils.tests_utils import general_example_test_filepath, create_test_log, general_example_test_filename, \
    create_test_encoding, create_test_labelling, general_example_train_filename, general_example_train_filepath, \
    create_test_job, create_test_predictive_model


class TestLabelSimpleIndex(TestCase):
    def setUp(self):
        self.train_log = get_log(create_test_log(log_name=general_example_train_filename,
                                                 log_path=general_example_train_filepath))
        self.test_log = get_log(create_test_log(log_name=general_example_test_filename,
                                                log_path=general_example_test_filepath))
        self.encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=1)

    def test_no_label(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2)
        labelling = create_test_labelling(label_type=LabelTypes.NO_LABEL.value)

        df, _ = encode_label_logs(
            self.test_log,
            self.test_log,
            create_test_job(
                encoding=encoding,
                labelling=labelling,
                predictive_model=create_test_predictive_model(
                    predictive_model=PredictiveModels.CLASSIFICATION.value)
            ))
        self.assertEqual(df.shape, (2, 3))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 1])

    def test_no_label_zero_padding(self):
        labelling = create_test_labelling(label_type=LabelTypes.NO_LABEL.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=10,
            padding=True,
            add_remaining_time=True)

        df, _ = encode_label_logs(self.test_log,
                                  self.test_log,
                                  create_test_job(
                                      encoding=encoding,
                                      labelling=labelling,
                                      predictive_model=create_test_predictive_model(
                                          predictive_model=PredictiveModels.CLASSIFICATION.value)
                                  ))
        self.assertEqual(df.shape, (2, 11))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 1, 1, 2, 0, 0, 1, 0, 'examine casually'])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 1, 0, 1, 3, 0, 0, 0, 0, 0])

    def test_remaining_time(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)

        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2)

        df, _ = encode_label_logs(self.test_log, self.test_log,
                                  create_test_job(
                                      encoding=encoding,
                                      labelling=labelling,
                                      predictive_model=create_test_predictive_model(
                                          predictive_model=PredictiveModels.CLASSIFICATION.value)
                                  ))
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 0])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 1, 0])

    def test_label_remaining_time_with_elapsed_time_custom_threshold(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=True,
            add_remaining_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2)
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value,
                                          threshold_type=ThresholdTypes.THRESHOLD_CUSTOM.value,
                                          threshold=40000)

        df, _ = encode_label_logs(self.test_log, self.test_log,
                                  create_test_job(
                                      encoding=encoding,
                                      labelling=labelling,
                                      predictive_model=create_test_predictive_model(
                                          predictive_model=PredictiveModels.CLASSIFICATION.value)
                                  ))
        self.assertEqual(df.shape, (2, 5))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'elapsed_time', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 0, 0])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 1, 0, 0])

    def test_remaining_time_zero_padding(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=10,
            padding=True)

        df, _ = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 13))
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5,
                             ['5', 1, 3, 2, 2, 2, 0, 0, 0, 0, 1, 1296240.0, 0])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4,
                             ['4', 52903968, 32171502, 17803069, 1149821, 72523760, 0, 0, 0, 0, 0, 520920.0, True])

    def test_next_activity(self):
        labelling = create_test_labelling(label_type=LabelTypes.NEXT_ACTIVITY.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2)

        df, _ = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 1])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 1, 2])

    def test_next_activity_zero_padding_elapsed_time(self):
        labelling = create_test_labelling(label_type=LabelTypes.NEXT_ACTIVITY.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=10,
            padding=True)

        df, _ = encode_label_logs(self.test_log, self.test_log,
                                  create_test_job(
                                      encoding=encoding,
                                      labelling=labelling,
                                      predictive_model=create_test_predictive_model(
                                          predictive_model=PredictiveModels.CLASSIFICATION.value)
                                  ))
        self.assertEqual(df.shape, (2, 13))
        self.assertTrue('elapsed_time' in df.columns.values.tolist())
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5,
                             ['5', 1, 3, 2, 2, 2, 0, 0, 0, 0, 1296240.0, 2])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4,
                             ['4', 52903968, 32171502, 17803069, 1149821, 72523760, 0, 0, 0, 0, 0, 520920.0, 0])

    def test_attribute_string(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2)
        labelling = create_test_labelling(label_type=LabelTypes.ATTRIBUTE_STRING.value, attribute_name='creator')

        df, _ = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 1])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 1, 1])

    def test_attribute_number(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2)
        labelling = create_test_labelling(label_type=LabelTypes.ATTRIBUTE_NUMBER.value, attribute_name='AMOUNT')

        df, _ = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 0])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 1, 0])

    def test_duration(self):
        labelling = create_test_labelling(label_type=LabelTypes.DURATION.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2)

        df, _ = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 4))
        self.assertListEqual(df.columns.values.tolist(), ['trace_id', 'prefix_1', 'prefix_2', 'label'])
        trace_5 = df[df.trace_id == '5'].iloc[0].values.tolist()
        self.assertListEqual(trace_5, ['5', 1, 2, 0])
        trace_4 = df[df.trace_id == '4'].iloc[0].values.tolist()
        self.assertListEqual(trace_4, ['4', 1, 1, 0])

    def test_add_executed_events(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=True,
            add_executed_events=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=1)

        df, _ = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 5))
        self.assertTrue('executed_events' in df.columns.values.tolist())
        self.assertListEqual(df['executed_events'].tolist(), [2, 2])

    def test_add_resources_used(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=True,
            add_resources_used=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=1)

        df, _ = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 5))
        self.assertTrue('resources_used' in df.columns.values.tolist())
        self.assertListEqual(df['resources_used'].tolist(), [1, 1])

    def test_add_new_traces(self):
        labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=True,
            add_new_traces=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=1)

        df, _ = encode_label_logs(self.test_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertEqual(df.shape, (2, 5))
        self.assertTrue('new_traces' in df.columns.values.tolist())
        self.assertListEqual(df['new_traces'].tolist(), [0, 0])
