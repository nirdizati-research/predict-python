import datetime
import pandas as pd

import numpy as np
from collections import namedtuple

from pandas.core.dtypes.common import is_datetime64tz_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Labeling methods
SIMPLE_INDEX = 'simpleIndex'
BOOLEAN = 'boolean'
FREQUENCY = 'frequency'
COMPLEX = 'complex'
LAST_PAYLOAD = 'lastPayload'

# Generation types
UP_TO = 'up_to'
ONLY_THIS = 'only'
ALL_IN_ONE = 'all_in_one'

# padding
ZERO_PADDING = 'zero_padding'
NO_PADDING = 'no_padding'
PADDING_VALUE = 0

# Encoding methods
LABEL_ENCODER = 'label_encoder'
ONE_HOT_ENCODER = 'one_hot'

ENCODING = LABEL_ENCODER

label_encoder = {}
encoder = {}
label_dict = {}


class EncodingContainer(namedtuple('EncodingContainer', ["method", "prefix_length", "padding", "generation_type"])):
    """Inner object describing encoding configuration.
    """

    def __new__(cls, method=SIMPLE_INDEX, prefix_length=1, padding=NO_PADDING,
                generation_type=ONLY_THIS):
        return super(EncodingContainer, cls).__new__(cls, method, prefix_length, padding, generation_type)

    def is_zero_padding(self):
        return self.padding == ZERO_PADDING

    def is_all_in_one(self):
        return self.generation_type == ALL_IN_ONE

    def is_boolean(self):
        return self.method == BOOLEAN

    def is_complex(self):
        return self.method == COMPLEX

    def encode(self, df):
        for column in df:
            if column in encoder:
                if ENCODING == LABEL_ENCODER:
                    df[column] = df[column].apply(lambda x: label_dict[column].get(x, PADDING_VALUE))
                elif ENCODING == ONE_HOT_ENCODER:
                    raise ValueError('Onehot encoder not yet implemented')
                    # values = np.array([ label_dict[column].get(x, label_dict[column][PADDING_VALUE]) for x in df[column] ])
                    # df[column] = np.array(encoder[column].transform(values.reshape(len(values), 1)).toarray())
                else:
                    raise ValueError('Please set the encoding technique!')

    def init_label_encoder(self, df):
        for column in df:
            if column != 'trace_id' and column != 'label':
                if df[column].dtype != int:
                    if ENCODING == LABEL_ENCODER:
                        if is_datetime64tz_dtype(pd.Series(df[column][df[column] != -1].values).dtype):
                            encoder[column] = LabelEncoder().fit(sorted(df[column]))
                        else:
                            encoder[column] = LabelEncoder().fit(
                                pd.concat([pd.Series([str(PADDING_VALUE)]), df[column].apply(lambda x: str(x))]))
                        classes = encoder[column].classes_
                        transforms = encoder[column].transform(encoder[column].classes_)
                        label_dict[column] = dict(zip(classes, transforms))
                    elif ENCODING == ONE_HOT_ENCODER:
                        raise ValueError('Onehot encoder not yet implemented')
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
