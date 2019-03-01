from django.test import TestCase

from src.encoding.complex_last_payload import complex
from src.encoding.encoding_container import EncodingContainer, ZERO_PADDING, ALL_IN_ONE
from src.encoding.models import ValueEncodings
from src.labelling.label_container import LabelContainer
from src.labelling.models import LabelTypes
from src.utils.event_attributes import unique_events, get_additional_columns
from src.utils.file_service import get_log
from src.utils.tests_utils import general_example_test_filepath


class Complex(TestCase):
    def setUp(self):
        self.log = get_log(general_example_test_filepath)
        self.event_names = unique_events(self.log)
        self.label = LabelContainer(add_elapsed_time=True)
        self.add_col = get_additional_columns(self.log)
        self.encoding = EncodingContainer(ValueEncodings.COMPLEX)

    def test_shape(self):
        encoding = EncodingContainer(ValueEncodings.COMPLEX, prefix_length=2)
        df = complex(self.log, self.label, encoding, self.add_col)

        self.assertEqual((2, 15), df.shape)
        headers = ['trace_id', 'AMOUNT', 'creator', 'prefix_1', 'Activity_1', 'Costs_1', 'Resource_1',
                   'org:resource_1', 'prefix_2', 'Activity_2', 'Costs_2', 'Resource_2', 'org:resource_2',
                   'elapsed_time', 'label']
        self.assertListEqual(headers, df.columns.values.tolist())

    def test_prefix1(self):
        df = complex(self.log, self.label, self.encoding, self.add_col)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ['5', '300', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Ellen',
                              'Ellen', 0.0, 1576440.0])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ['4', '100', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Pete',
                              'Pete', 0.0, 520920.0])

    def test_prefix1_no_label(self):
        label = LabelContainer(LabelTypes.NO_LABEL.value)
        df = complex(self.log, label, self.encoding, self.add_col)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ['5', '300', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Ellen',
                              'Ellen'])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ['4', '100', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Pete',
                              'Pete'])

    def test_prefix1_no_elapsed_time(self):
        df = complex(self.log, LabelContainer(), self.encoding, self.add_col)

        row1 = df[(df.trace_id == '5')].iloc[0].tolist()
        self.assertListEqual(row1,
                             ['5', '300', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Ellen',
                              'Ellen', 1576440.0])
        row2 = df[(df.trace_id == '4')].iloc[0].tolist()
        self.assertListEqual(row2,
                             ['4', '100', 'Fluxicon Nitro', 'register request', 'register request', '50', 'Pete',
                              'Pete', 520920.0])

    def test_prefix2(self):
        encoding = EncodingContainer(ValueEncodings.COMPLEX, prefix_length=2)
        df = complex(self.log, self.label, encoding, self.add_col)

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
        encoding = EncodingContainer(ValueEncodings.COMPLEX, prefix_length=5)
        df = complex(self.log, self.label, encoding, self.add_col)

        self.assertEqual(df.shape, (2, 30))
        self.assertFalse(df.isnull().values.any())

    def test_prefix10(self):
        encoding = EncodingContainer(ValueEncodings.COMPLEX, prefix_length=10)
        df = complex(self.log, self.label, encoding, self.add_col)

        self.assertEqual(df.shape, (1, 55))
        self.assertFalse(df.isnull().values.any())

    def test_prefix10_zero_padding(self):
        encoding = EncodingContainer(ValueEncodings.COMPLEX, prefix_length=10, padding=ZERO_PADDING)
        df = complex(self.log, self.label, encoding, self.add_col)

        self.assertEqual(df.shape, (2, 55))
        self.assertFalse(df.isnull().values.any())

    def test_prefix10_all_in_one(self):
        encoding = EncodingContainer(ValueEncodings.COMPLEX, prefix_length=10, generation_type=ALL_IN_ONE)
        df = complex(self.log, self.label, encoding, self.add_col)

        self.assertEqual(df.shape, (10, 55))
        self.assertFalse(df.isnull().values.any())

    def test_prefix10_zero_padding_all_in_one(self):
        encoding = EncodingContainer(ValueEncodings.COMPLEX, prefix_length=10, padding=ZERO_PADDING,
                                     generation_type=ALL_IN_ONE)
        df = complex(self.log, self.label, encoding, self.add_col)

        self.assertEqual(df.shape, (15, 55))
        self.assertFalse(df.isnull().values.any())
