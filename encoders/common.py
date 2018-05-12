import hashlib

from core.constants import *
from encoders.boolean_frequency import frequency
from encoders.complex_last_payload import complex, last_payload
from encoders.label_container import *
from log_util.event_attributes import unique_events2
from .boolean_frequency import boolean
from .simple_index import simple_index


def encode_label_logs(training_log: list, test_log: list, encoding_type: str, job_type: str, label: LabelContainer,
                      prefix_length=1, zero_padding=False, additional_columns=None):
    """Encodes and labels test set and training set as data frames

    :param prefix_length Applies to all
    :param additional_columns Global trace attributes for complex and last payload encoding
    :returns training_df, test_df
    """
    event_names = unique_events2(training_log, test_log)
    training_df = encode_label_log(training_log, encoding_type, job_type, label, event_names,
                                   prefix_length=prefix_length, zero_padding=zero_padding,
                                   additional_columns=additional_columns)
    test_df = encode_label_log(test_log, encoding_type, job_type, label, event_names, prefix_length=prefix_length,
                               zero_padding=zero_padding, additional_columns=additional_columns)
    return training_df, test_df


def encode_label_log(run_log: list, encoding_type: str, job_type: str, label: LabelContainer, event_names=None,
                     prefix_length=1, zero_padding=False, additional_columns=None):
    encoded_log = encode_log(run_log, encoding_type, label, prefix_length, event_names, zero_padding,
                             additional_columns)

    # Convert strings to number
    if label.type == ATTRIBUTE_NUMBER:
        encoded_log['label'] = encoded_log['label'].apply(lambda x: float(x))

    # converts string values to in
    if job_type != LABELLING:
        # Labelling has no need for this encoding
        categorical_encode(encoded_log)
    # Regression only has remaining_time or number atr as label
    if job_type == REGRESSION:
        return encoded_log
    # Post processing
    if label.type == REMAINING_TIME or label.type == ATTRIBUTE_NUMBER or label.type == DURATION:
        return label_boolean(encoded_log, label)
    return encoded_log


def encode_log(run_log: list, encoding_type: str, label: LabelContainer, prefix_length=1, event_names=None,
               zero_padding=False, additional_columns=None, encoding=None):
    """Encodes test set and training set as data frames

    :param prefix_length consider up to this event in log
    :param additional_columns Global trace attributes for complex and last payload encoding
    :param zero_padding If log shorter than prefix_length, weather to skip or pad with 0 up to prefix_length
    :returns training_df, test_df
    """
    if prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    run_df = None
    if encoding_type == SIMPLE_INDEX:
        run_df = simple_index(run_log, label, encoding)
    elif encoding_type == BOOLEAN:
        run_df = boolean(run_log, event_names, label, prefix_length=prefix_length, zero_padding=zero_padding)
    elif encoding_type == FREQUENCY:
        run_df = frequency(run_log, event_names, label, prefix_length=prefix_length, zero_padding=zero_padding)
    elif encoding_type == COMPLEX:
        run_df = complex(run_log, label, additional_columns, prefix_length=prefix_length,
                         zero_padding=zero_padding)
    elif encoding_type == LAST_PAYLOAD:
        run_df = last_payload(run_log, label, additional_columns, prefix_length=prefix_length,
                              zero_padding=zero_padding)
    return run_df


def label_boolean(df, label: LabelContainer):
    """Label a numeric attribute as True or False based on threshold
    This is essentially a Fast/Slow classification without string labels
    By default use mean of label value
    True if under threshold value
    """
    if label.threshold_type == THRESHOLD_MEAN:
        threshold_ = df['label'].mean()
    else:
        threshold_ = float(label.threshold)
    df['label'] = df['label'] < threshold_
    return df


def categorical_encode(df):
    """Encodes every column except trace_id and label as int

    Encoders module puts event name in cell, which can't be used by machine learning methods directly.
    """
    for column in df.columns:
        if column == 'trace_id':
            continue
        elif df[column].dtype == type(str):
            df[column] = df[column].map(lambda s: convert(s))
    return df


def convert(s):
    if isinstance(s, float) or isinstance(s, int):
        return s
    if s is None:
        # Next activity resources
        s = '0'
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8
