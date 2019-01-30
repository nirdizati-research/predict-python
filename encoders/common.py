import hashlib

from pandas import DataFrame

from core.constants import LABELLING, REGRESSION
from encoders.boolean_frequency import frequency, boolean
from encoders.complex_last_payload import complex, last_payload
from encoders.encoding_container import EncodingContainer, SIMPLE_INDEX, BOOLEAN, FREQUENCY, COMPLEX, LAST_PAYLOAD
from encoders.label_container import *
from utils.event_attributes import unique_events2, unique_events
from .simple_index import simple_index


def encode_label_logs(training_log: list, test_log: list, encoding: EncodingContainer, job_type: str,
                      label: LabelContainer, additional_columns=None):
    event_names = unique_events2(training_log, test_log)
    training_df = encode_label_log(training_log, encoding, job_type, label, event_names=event_names,
                                   additional_columns=additional_columns)
    test_df = encode_label_log(test_log, encoding, job_type, label, event_names=event_names,
                               additional_columns=additional_columns)
    return training_df, test_df


def encode_label_log(run_log: list, encoding: EncodingContainer, job_type: str, label: LabelContainer, event_names=None,
                     additional_columns=None):
    if event_names is None:
        event_names = unique_events(run_log)

    encoded_log = _encode_log(run_log, encoding, label, event_names, additional_columns)

    # Convert strings to number
    if label.type == ATTRIBUTE_NUMBER:
        encoded_log['label'] = encoded_log['label'].apply(lambda x: float(x))

    # converts string values to in
    if job_type != LABELLING:
        # Labelling has no need for this encoding
        _categorical_encode(encoded_log)
    # Regression only has remaining_time or number atr as label
    if job_type == REGRESSION:
        # Remove last events as worse for prediction
        # TODO filter out 0 labels. Doing it here means runtime errors for regression
        # if label.type == REMAINING_TIME:
        #     encoded_log = encoded_log.loc[encoded_log['label'] != 0.0]
        return encoded_log
    # Post processing
    if label.type == REMAINING_TIME or label.type == ATTRIBUTE_NUMBER or label.type == DURATION:
        return _label_boolean(encoded_log, label)
    return encoded_log


def _encode_log(run_log: list, encoding: EncodingContainer, label: LabelContainer, event_names=None,
                additional_columns=None):
    if encoding.prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    if encoding.method == SIMPLE_INDEX:
        run_df = simple_index(run_log, label, encoding)
    elif encoding.method == BOOLEAN:
        run_df = boolean(run_log, event_names, label, encoding)
    elif encoding.method == FREQUENCY:
        run_df = frequency(run_log, event_names, label, encoding)
    elif encoding.method == COMPLEX:
        run_df = complex(run_log, label, encoding, additional_columns)
    elif encoding.method == LAST_PAYLOAD:
        run_df = last_payload(run_log, label, encoding, additional_columns)
    else:
        raise ValueError("Unknown encoding method {}".format(encoding.method))
    return run_df


def _label_boolean(df: DataFrame, label: LabelContainer):
    """Label a numeric attribute as True or False based on threshold
    This is essentially a Fast/Slow classification without string labels
    By default use mean of label value
    True if under threshold value
    """
    if label.threshold_type == THRESHOLD_MEAN:
        threshold = df['label'].mean()
    else:
        threshold = float(label.threshold)
    df['label'] = df['label'] < threshold
    return df


def _categorical_encode(df):
    """Encodes every column except trace_id and label as int

    Encoders module puts event name in cell, which can't be used by machine learning methods directly.
    """
    for column in df.columns:
        if column == 'trace_id':
            continue
        elif df[column].dtype == type(str):
            df[column] = df[column].map(lambda s: _convert(s))
    return df


def _convert(s):
    if isinstance(s, float) or isinstance(s, int):
        return s
    if s is None:
        # Next activity resources
        s = '0'
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8
