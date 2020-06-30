from django.test import TestCase

from src.encoding.complex_last_payload import complex
from src.encoding.models import ValueEncodings, TaskGenerationTypes
from src.labelling.label_container import LabelContainer
from src.labelling.models import LabelTypes
from src.utils.event_attributes import unique_events, get_additional_columns
from src.logs.log_service import get_log
from src.utils.tests_utils import general_example_test_filepath_xes, create_test_log, general_example_test_filename, \
    create_test_encoding, create_test_labelling


class Complex(TestCase):
    def setUp(self):
        self.log = get_log(create_test_log(log_name=general_example_test_filename,
                                           log_path=general_example_test_filepath_xes))
        self.event_names = unique_events(self.log)
        self.add_col = get_additional_columns(self.log)
        self.encoding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=1)
        self.labelling = create_test_labelling(label_type=LabelTypes.REMAINING_TIME.value)

    def test_shape(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2)
        df = complex(self.log, self.labelling, encoding, self.add_col)

        self.assertEqual((2, 15), df.shape)
        headers = ['trace_id', 'AMOUNT', 'creator', 'prefix_1', 'Activity_1', 'Costs_1', 'Resource_1',
                   'org:resource_1', 'prefix_2', 'Activity_2', 'Costs_2', 'Resource_2', 'org:resource_2',
                   'elapsed_time', 'label']
        self.assertListEqual(headers, df.columns.values.tolist())

    def test_prefix1(self):
        df = complex(self.log, self.labelling, self.encoding, self.add_col)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ['5', '300', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Ellen',
                              'Ellen', 0.0, 1576440.0])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ['4', '100', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Pete',
                              'Pete', 0.0, 520920.0])

    def test_prefix1_no_label(self):
        labelling = create_test_labelling(label_type=LabelTypes.NO_LABEL.value)
        df = complex(self.log, labelling, self.encoding, self.add_col)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ['5', '300', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Ellen',
                              'Ellen'])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ['4', '100', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Pete',
                              'Pete'])

    def test_prefix1_no_elapsed_time(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=1)
        df = complex(self.log, LabelContainer(), encoding, self.add_col)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ['5', '300', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Ellen',
                              'Ellen', 1576440.0])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ['4', '100', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Pete',
                              'Pete', 520920.0])

    def test_prefix2(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=2)
        df = complex(self.log, self.labelling, encoding, self.add_col)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ['5', '300', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Ellen',
                              'Ellen', 'examine casually', 'examine casually', '400', 'Mike', 'Mike', 90840.0,
                              1485600.0])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ['4', '100', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Pete',
                              'Pete', 'check ticket', 'check ticket', '100', 'Mike', 'Mike', 75840.0, 445080.0])

    def test_prefix5(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=5)
        df = complex(self.log, self.labelling, encoding, self.add_col)

        self.assertEqual(df.shape, (2, 30))
        self.assertFalse(df.isnull().values.any())

    def test_prefix10(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=10)
        df = complex(self.log, self.labelling, encoding, self.add_col)

        self.assertEqual(df.shape, (1, 55))
        self.assertFalse(df.isnull().values.any())

    def test_prefix10_zero_padding(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ONLY_THIS.value,
            prefix_length=10,
            padding=True)
        df = complex(self.log, self.labelling, encoding, self.add_col)

        self.assertEqual(df.shape, (2, 55))
        self.assertFalse(df.isnull().values.any())

    def test_prefix10_all_in_one(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ALL_IN_ONE.value,
            prefix_length=10
        )
        df = complex(self.log, self.labelling, encoding, self.add_col)

        self.assertEqual(df.shape, (10, 55))
        self.assertFalse(df.isnull().values.any())

    def test_prefix10_zero_padding_all_in_one(self):
        encoding = create_test_encoding(
            value_encoding=ValueEncodings.COMPLEX.value,
            add_elapsed_time=True,
            task_generation_type=TaskGenerationTypes.ALL_IN_ONE.value,
            prefix_length=10,
            padding=True)
        df = complex(self.log, self.labelling, encoding, self.add_col)

        self.assertEqual(df.shape, (15, 55))
        self.assertFalse(df.isnull().values.any())
