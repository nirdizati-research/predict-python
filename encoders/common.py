from core.constants import *
from encoders.boolean_frequency import frequency
from encoders.complex_last_payload import complex, last_payload
from encoders.label_container import *
from .boolean_frequency import boolean
from .log_util import unique_events2
from .simple_index import simple_index


def encode_label_logs(training_log: list, test_log: list, encoding_type: str, job_type: str, label: LabelContainer,
                      prefix_length=1, zero_padding=False):
    """Encodes and labelstest set and training set as data frames

    :param prefix_length only for SIMPLE_INDEX, COMPLEX, LAST_PAYLOAD
    :returns training_df, test_df
    """
    event_names = unique_events2(training_log, test_log)
    training_df = encode_label_log(training_log, encoding_type, job_type, label, event_names,
                                   prefix_length=prefix_length, zero_padding=zero_padding)
    test_df = encode_label_log(test_log, encoding_type, job_type, label, event_names, prefix_length=prefix_length,
                               zero_padding=zero_padding)
    return training_df, test_df


def encode_label_log(run_log: list, encoding_type: str, job_type: str, label: LabelContainer, event_names=None,
                     prefix_length=1, zero_padding=False):
    encoded_log = encode_log(run_log, encoding_type, label, prefix_length, event_names, zero_padding)

    # Convert strings to number
    if label.type == ATTRIBUTE_NUMBER:
        encoded_log['label'] = encoded_log['label'].apply(lambda x: float(x))

    # Regression only has remaining_time as label
    if job_type == CLASSIFICATION:
        # Post processing
        if label.type == REMAINING_TIME or label.type == ATTRIBUTE_NUMBER:
            return label_boolean(encoded_log, label)
    return encoded_log


# def encode_logs(training_log: list, test_log: list, label: LabelContainer, encoding_type: str, job_type: str,
#                 prefix_length=1, zero_padding=False):
#     """Encodes test set and training set as data frames
#
#     :param prefix_length only for SIMPLE_INDEX, COMPLEX, LAST_PAYLOAD
#     :returns training_df, test_df
#     """
#     event_names = unique_events2(training_log, test_log)
#     training_df = encode_log(training_log, encoding_type, label, prefix_length, event_names, zero_padding=zero_padding)
#     test_df = encode_log(test_log, encoding_type, label, prefix_length, event_names, zero_padding=zero_padding)
#     return training_df, test_df


def encode_log(run_log: list, encoding_type: str, label: LabelContainer, prefix_length=1, event_names=None,
               zero_padding=False):
    """Encodes test set and training set as data frames

    :param prefix_length only for SIMPLE_INDEX, COMPLEX, LAST_PAYLOAD
    :returns training_df, test_df
    """
    run_df = None
    if encoding_type == SIMPLE_INDEX:
        run_df = simple_index(run_log, event_names, label, prefix_length=prefix_length, zero_padding=zero_padding)
    elif encoding_type == BOOLEAN:
        run_df = boolean(run_log, event_names, label)
    elif encoding_type == FREQUENCY:
        run_df = frequency(run_log, event_names, label)
    elif encoding_type == COMPLEX:
        run_df = complex(run_log, event_names, label, prefix_length=prefix_length,
                         zero_padding=zero_padding)
    elif encoding_type == LAST_PAYLOAD:
        run_df = last_payload(run_log, event_names, label, prefix_length=prefix_length,
                              zero_padding=zero_padding)
    return run_df


def label_boolean(df, label: LabelContainer):
    """Label a numeric attribute as True or False based on threshold
    By default use mean of label value
    True if under threshold value
    """
    if label.threshold_type == THRESHOLD_MEAN:
        threshold_ = df['label'].mean()
    else:
        threshold_ = float(label.threshold)
    df['label'] = df['label'] < threshold_
    return df
