from django.test import TestCase
from pandas import DataFrame

from src.encoding.encoder import Encoder
from src.encoding.models import DataEncodings
from src.utils.tests_utils import create_test_encoding


class TestEncoder(TestCase):
    def setUp(self):
        self.df = DataFrame({
            'literal_feature': [ str(item) for item in ['a', 'b', None] ],
            'numeric_feature':  [ str(item) for item in [.1, 1, -.99] ],
            'misc_feature':  [ str(item) for item in ['a', None, -.99] ]
        })
        self.how_it_should_be = DataFrame({
            'literal_feature': [2, 3, 1],
            'numeric_feature': [2, 3, 0],
            'misc_feature': [3, 2, 0]
        })
        self.encoding = create_test_encoding()

    def test_encoder(self):
        encoder = Encoder(df=self.df, encoding=self.encoding)

        self.assertIsNotNone(encoder._encoder)
        self.assertIsNotNone(encoder._label_dict)
        self.assertIsNotNone(encoder._label_dict_decoder)

    def test_encode(self):
        encoder = Encoder(df=self.df, encoding=self.encoding)
        encoded_df = self.df.copy()
        encoder.encode(df=encoded_df, encoding=self.encoding)

        self.assertDictEqual(self.how_it_should_be.to_dict(), encoded_df.to_dict())

    def test_decode(self):
        encoder = Encoder(df=self.df, encoding=self.encoding)
        encoded_df = self.df.copy()
        encoder.encode(df=encoded_df, encoding=self.encoding)
        encoder.decode(df=encoded_df, encoding=self.encoding)
        self.assertDictEqual(self.df.to_dict(), encoded_df.to_dict())

    def test_repeated_encode_decode(self):
        encoder = Encoder(df=self.df, encoding=self.encoding)
        encoded_df = self.df.copy()
        encoder.encode(df=encoded_df, encoding=self.encoding)
        encoder.decode(df=encoded_df, encoding=self.encoding)
        encoder.encode(df=encoded_df, encoding=self.encoding)
        encoder.decode(df=encoded_df, encoding=self.encoding)
        self.assertDictEqual(self.df.to_dict(), encoded_df.to_dict())

    def test_NotImplementedException_init_encoder(self):
        try:
            Encoder(
                df=self.df,
                encoding=create_test_encoding(data_encoding=DataEncodings.ONE_HOT_ENCODER.value)
            )
        except NotImplementedError:
            pass

    def test_NotImplementedException_encode(self):
        try:
            encoder = Encoder(df=self.df, encoding=self.encoding)
            encoder.encode(
                df=self.df,
                encoding=create_test_encoding(data_encoding=DataEncodings.ONE_HOT_ENCODER.value)
            )
        except NotImplementedError:
            pass

    def test_NotImplementedException_decode(self):
        try:
            encoder = Encoder(df=self.df, encoding=self.encoding)
            encoder.decode(
                df=self.df,
                encoding=create_test_encoding(data_encoding=DataEncodings.ONE_HOT_ENCODER.value)
            )
        except NotImplementedError:
            pass

    def test_ValueError_init_encoder(self):
        try:
            Encoder(
                df=self.df,
                encoding=create_test_encoding(data_encoding='None')
            )
        except ValueError:
            pass

    def test_ValueError_encode(self):
        try:
            encoder = Encoder(df=self.df, encoding=self.encoding)
            encoder.encode(
                df=self.df,
                encoding=create_test_encoding(data_encoding='None')
            )
        except ValueError:
            pass

    def test_ValueError_decode(self):
        try:
            encoder = Encoder(df=self.df, encoding=self.encoding)
            encoder.decode(
                df=self.df,
                encoding=create_test_encoding(data_encoding='None')
            )
        except ValueError:
            pass
