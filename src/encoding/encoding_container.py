from collections import namedtuple

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

# Generation types
from src.encoding.models import ValueEncodings, DataEncodings

UP_TO = 'up_to'
ONLY_THIS = 'only'
ALL_IN_ONE = 'all_in_one'

# padding
ZERO_PADDING = 'zero_padding'
NO_PADDING = 'no_padding'
PADDING_VALUE = 0

# Encoding methods

label_encoder = {}
encoder = {}
label_dict = {}

ENCODING = DataEncodings.LABEL_ENCODER

PADDINGS = [ZERO_PADDING, NO_PADDING]

TIME_SERIES_PREDICTION_PADDINGS = [ZERO_PADDING]


class EncodingContainer(namedtuple('EncodingContainer', ['method', 'prefix_length', 'padding', 'generation_type'])):
    """Inner object describing encoding configuration.
    """

    def __new__(cls, method: ValueEncodings = ValueEncodings.SIMPLE_INDEX, prefix_length: int = 1,
                padding: str = NO_PADDING,
                generation_type: str = ONLY_THIS):
        # noinspection PyArgumentList
        return super(EncodingContainer, cls).__new__(cls, method, prefix_length, padding, generation_type)

    def is_zero_padding(self) -> bool:
        return self.padding == ZERO_PADDING

    def is_all_in_one(self) -> bool:
        return self.generation_type == ALL_IN_ONE

    def is_boolean(self) -> bool:
        return self.method == ValueEncodings.BOOLEAN

    def is_complex(self) -> bool:
        return self.method == ValueEncodings.COMPLEX

    @staticmethod
    def encode(df: DataFrame) -> None:
        for column in df:
            if column in encoder:
                if ENCODING == DataEncodings.LABEL_ENCODER:
                    df[column] = df[column].apply(lambda x: label_dict[column].get(x, PADDING_VALUE))
                elif ENCODING == DataEncodings.ONE_HOT_ENCODER:
                    raise NotImplementedError('Onehot encoder not yet implemented')
                    # values = np.array([ label_dict[column].get(x, label_dict[column][PADDING_VALUE]) for
                    # x in df[column] ])
                    # df[column] = np.array(encoder[column].transform(values.reshape(len(values), 1)).toarray())
                else:
                    raise ValueError('Please set the encoding technique!')

    @staticmethod
    def init_label_encoder(df: DataFrame) -> None:
        for column in df:
            if column != 'trace_id':
                if df[column].dtype != int or (df[column].dtype == int and pd.np.any(df[column] < 0)):
                    if ENCODING == DataEncodings.LABEL_ENCODER:
                        encoder[column] = LabelEncoder().fit(
                            sorted(pd.concat([pd.Series([str(PADDING_VALUE)]), df[column].apply(lambda x: str(x))])))
                        classes = encoder[column].classes_
                        transforms = encoder[column].transform(classes)
                        label_dict[column] = dict(zip(classes, transforms))
                    elif ENCODING == DataEncodings.ONE_HOT_ENCODER:
                        raise NotImplementedError('Onehot encoder not yet implemented')
                        # label_encoder[column] = LabelEncoder().fit(df[column])
                        # classes = label_encoder[column].classes_
                        # transforms = label_encoder[column].transform(label_encoder[column].classes_)
                        # label_dict[column] = dict(zip(classes, transforms))
                        # label_dict[column][PADDING_VALUE] = -1
                        # if min(transforms) < label_dict[column][PADDING_VALUE]:
                        #     print('-1 is not a proper value as padding, switching to min-1')
                        #     label_dict[column][PADDING_VALUE] = min(transforms) - 1
                        # encoder[column] = OneHotEncoder(handle_unknown='ignore').fit(
                        #     label_encoder[column].transform(df[column]).reshape(len(df[column]), 1))
                    else:
                        raise ValueError('Please set the encoding technique!')
