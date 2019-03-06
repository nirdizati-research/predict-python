from django.test import TestCase

from src.encoding.common import encode_label_logs, LabelTypes
from src.encoding.models import TaskGenerationTypes, ValueEncodings
from src.encoding.simple_index import simple_index
from src.predictive_model.models import PredictiveModels
from src.utils.file_service import get_log
from src.utils.tests_utils import general_example_filepath, general_example_train_filepath, \
    general_example_test_filepath, general_example_test_filename, create_test_log, general_example_train_filename, \
    create_test_predictive_model, create_test_job, create_test_encoding, create_test_labelling, general_example_filename
from django.test import TestCase

from src.encoding.common import encode_label_logs, LabelTypes
from src.encoding.models import TaskGenerationTypes, ValueEncodings
from src.encoding.simple_index import simple_index
from src.predictive_model.models import PredictiveModels
from src.utils.file_service import get_log
from src.utils.tests_utils import general_example_filepath, general_example_train_filepath, \
    general_example_test_filepath, general_example_test_filename, create_test_log, general_example_train_filename, \
    create_test_predictive_model, create_test_job, create_test_encoding, create_test_labelling, general_example_filename


class TestSplitLogExample(TestCase):
    def setUp(self):
        self.test_log = get_log(create_test_log(log_name=general_example_test_filename,
                                                log_path=general_example_test_filepath))
        self.training_log = get_log(create_test_log(log_name=general_example_train_filename,
                                                    log_path=general_example_train_filepath))
        self.labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        self.encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=1)

    def test_shape_training(self):
        training_df, test_df = encode_label_logs(self.training_log, self.test_log, create_test_job(
            encoding=self.encoding,
            labelling=self.labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assert_shape(training_df, (4, 4))
        self.assert_shape(test_df, (2, 4))

    def assert_shape(self, dataframe, shape):
        self.assertIn("trace_id", dataframe.columns.values)
        self.assertIn("label", dataframe.columns.values)
        self.assertIn("elapsed_time", dataframe.columns.values)
        self.assertIn("prefix_1", dataframe.columns.values)
        self.assertEqual(shape, dataframe.shape)

    def test_prefix_length_training(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=3)
        training_df, test_df = encode_label_logs(self.training_log, self.test_log, create_test_job(
            encoding=encoding,
            labelling=self.labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        self.assertIn("prefix_1", training_df.columns.values)
        self.assertIn("prefix_2", training_df.columns.values)
        self.assertIn("prefix_3", training_df.columns.values)
        self.assertEqual((4, 6), training_df.shape)
        self.assertEqual((2, 6), test_df.shape)

        row = training_df[(training_df.trace_id == '3')].iloc[0]
        self.assertEqual(1, row.prefix_1)
        self.assertEqual(2, row.prefix_2)
        self.assertEqual(1, row.prefix_3)
        self.assertEqual(False, row.label)
        self.assertEqual(0, row.elapsed_time)

    def test_row_test(self):
        training_df, test_df = encode_label_logs(self.training_log, self.test_log, create_test_job(
            encoding=self.encoding,
            labelling=self.labelling,
            predictive_model=create_test_predictive_model(
                predictive_model=PredictiveModels.CLASSIFICATION.value)
        ))
        row = test_df[(test_df.trace_id == '4')].iloc[0]

        self.assertEqual(1, row.prefix_1)
        self.assertEqual(0, row.elapsed_time)
        self.assertEqual(0, row.label)

    def test_prefix0(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.FREQUENCY.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=0)
        self.assertRaises(ValueError,
                          encode_label_logs, self.training_log, self.test_log, create_test_job(
                encoding=encoding,
                labelling=self.labelling,
                predictive_model=create_test_predictive_model(
                    predictive_model=PredictiveModels.CLASSIFICATION.value)
            ))


class TestGeneralTest(TestCase):
    """Making sure it actually works"""

    def setUp(self):
        self.log = get_log(create_test_log(log_name=general_example_test_filename,
                                           log_path=general_example_test_filepath))
        self.labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        self.encoding = create_test_encoding(
            value_encoding=ValueEncodings.SIMPLE_INDEX.value,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            add_elapsed_time=True,
            prefix_length=1)

    def test_header(self):
        df = simple_index(self.log, self.labelling, self.encoding)

        self.assertIn("trace_id", df.columns.values)
        self.assertIn("label", df.columns.values)
        self.assertIn("elapsed_time", df.columns.values)
        self.assertIn("prefix_1", df.columns.values)

    def test_prefix1(self):
        df = simple_index(self.log, self.labelling, self.encoding)

        self.assertEqual(df.shape, (2, 4))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 'register request', 0.0, 1576440.0], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 'register request', 0.0, 520920.0], row2.values.tolist())

    def test_prefix1_no_label(self):
        df = simple_index(self.log, create_test_labelling(label_type=LabelTypes.NO_LABEL.value), self.encoding)

        self.assertEqual(df.shape, (2, 2))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 'register request'], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 'register request'], row2.values.tolist())

    def test_prefix1_no_elapsed_time(self):
        label = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.FREQUENCY.value,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=1)
        df = simple_index(self.log, label, encoding)

        self.assertEqual(df.shape, (2, 3))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 'register request', 1576440.0], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 'register request', 520920.0], row2.values.tolist())

    def test_prefix2(self):
        df = simple_index(self.log, self.labelling, create_test_encoding(
            value_encoding=ValueEncodings.FREQUENCY.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2))

        self.assertEqual(df.shape, (2, 5))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(['5', 'register request', 'examine casually', 90840.0, 1485600.0], row1.values.tolist())
        row2 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(['4', 'register request', 'check ticket', 75840.0, 445080.0], row2.values.tolist())

    def test_prefix5(self):
        df = simple_index(self.log, self.labelling, create_test_encoding(
            value_encoding=ValueEncodings.FREQUENCY.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=5))

        self.assertEqual(df.shape, (2, 8))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(
            ['5', 'register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request', 458160.0,
             1118280.0], row1.values.tolist())
        self.assertFalse(df.isnull().values.any())

    def test_prefix10(self):
        df = simple_index(self.log, self.labelling, create_test_encoding(
            value_encoding=ValueEncodings.FREQUENCY.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=10))

        self.assertEqual(df.shape, (1, 13))
        row1 = df[df.trace_id == '5'].iloc[0]
        self.assertListEqual(
            ['5', 'register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request',
             'check ticket', 'examine casually', 'decide', 'reinitiate request', 'examine casually', 1296240.0,
             280200.0], row1.values.tolist())

    def test_prefix10_padding(self):
        df = simple_index(self.log, self.labelling, create_test_encoding(
            value_encoding=ValueEncodings.FREQUENCY.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=10, padding=True))

        self.assertEqual(df.shape, (2, 13))
        row1 = df[df.trace_id == '4'].iloc[0]
        self.assertListEqual(
            ['4', 'register request', 'check ticket', 'examine thoroughly', 'decide', 'reject request', 0, 0, 0,
             0, 0, 520920.0, 0.0], row1.values.tolist())
        self.assertFalse(df.isnull().values.any())

    def test_prefix10_all_in_one(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.FREQUENCY.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ALL_IN_ONE.value,
            prefix_length=10)
        df = simple_index(self.log, self.labelling, encoding)

        self.assertEqual(df.shape, (10, 13))
        row1 = df[df.trace_id == '5'].iloc[9]
        self.assertListEqual(
            ['5', 'register request', 'examine casually', 'check ticket', 'decide', 'reinitiate request',
             'check ticket', 'examine casually', 'decide', 'reinitiate request', 'examine casually', 1296240.0,
             280200.0], row1.values.tolist())
        self.assertFalse(df.isnull().values.any())

    def test_prefix10_padding_all_in_one(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.FREQUENCY.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ALL_IN_ONE.value,
            prefix_length=10,
            padding=True)
        df = simple_index(self.log, self.labelling, encoding)

        self.assertEqual(df.shape, (15, 13))
        row1 = df[df.trace_id == '4'].iloc[4]
        self.assertListEqual(
            ['4', 'register request', 'check ticket', 'examine thoroughly', 'decide', 'reject request', 0, 0, 0,
             0, 0, 520920.0, 0.0], row1.values.tolist())
        self.assertFalse(df.isnull().values.any())

    def test_eval(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.FREQUENCY.value,
            task_generation_type=TaskGenerationTypes.ALL_IN_ONE.value,
            add_elapsed_time=True,
            prefix_length=12,
            padding=True)
        df = simple_index(
            get_log(create_test_log(log_path=general_example_filepath, log_name=general_example_filename)),
            create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value), encoding)

        self.assertEqual(df.shape, (41, 15))
        row1 = df[df.trace_id == '4'].iloc[4]
        self.assertListEqual(
            ['4', 'register request', 'check ticket', 'examine thoroughly', 'decide', 'reject request', 0, 0, 0,
             0, 0, 0, 0, 520920.0, 0.0], row1.values.tolist())
        self.assertFalse(df.isnull().values.any())
