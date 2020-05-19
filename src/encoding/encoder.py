import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from src.encoding.models import Encoding, DataEncodings

UP_TO = 'up_to'
ONLY_THIS = 'only'
ALL_IN_ONE = 'all_in_one'

# padding
ZERO_PADDING = 'zero_padding'
NO_PADDING = 'no_padding'
PADDING_VALUE = 0

PREFIX_ = 'prefix_'


class Encoder:
    def __init__(self, df: DataFrame, encoding: Encoding):
        self._encoder = {}
        self._label_dict = {}
        self._label_dict_decoder = {}
        self._init_encoder(df, encoding)

    def _init_encoder(self, df: DataFrame, encoding: Encoding):
        for column in df:
            if column != 'trace_id':
                if df[column].dtype != int or (df[column].dtype == int and np.any(df[column] < 0)):
                    if encoding.data_encoding == DataEncodings.LABEL_ENCODER.value:
                        self._encoder[column] = LabelEncoder().fit(
                            sorted(pd.concat([pd.Series([str(PADDING_VALUE)]), df[column].apply(lambda x: str(x))])))
                        classes = self._encoder[column].classes_
                        transforms = self._encoder[column].transform(classes)
                        self._label_dict[column] = dict(zip(classes, transforms))
                        self._label_dict_decoder[column] = dict(zip(transforms, classes))
                    elif encoding.data_encoding == DataEncodings.ONE_HOT_ENCODER.value:
                        raise NotImplementedError('Onehot encoder not yet implemented')
                    else:
                        raise ValueError('Please set the encoding technique!')

    def encode(self, df: DataFrame, encoding: Encoding) -> None:
        for column in df:
            if column in self._encoder:
                if encoding.data_encoding == DataEncodings.LABEL_ENCODER.value:
                    df[column] = df[column].apply(lambda x: self._label_dict[column].get(str(x), PADDING_VALUE))
                elif encoding.data_encoding == DataEncodings.ONE_HOT_ENCODER.value:
                    raise NotImplementedError('Onehot encoder not yet implemented')
                else:
                    raise ValueError('Please set the encoding technique!')

    def decode(self, df: DataFrame, encoding: Encoding) -> None:
        for column in df:
            if column in self._encoder:
                if encoding.data_encoding == DataEncodings.LABEL_ENCODER.value:
                    df[column] = df[column].apply(lambda x: self._label_dict_decoder[column].get(x, PADDING_VALUE))
                elif encoding.data_encoding == DataEncodings.ONE_HOT_ENCODER.value:
                    raise NotImplementedError('Onehot encoder not yet implemented')
                else:
                    raise ValueError('Please set the encoding technique!')
