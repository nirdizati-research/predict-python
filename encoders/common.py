import hashlib

from core.constants import LABELLING, REGRESSION
from encoders.boolean_frequency import frequency, boolean
from encoders.complex_last_payload import complex, last_payload
from encoders.encoding_container import EncodingContainer, SIMPLE_INDEX, BOOLEAN, FREQUENCY, COMPLEX, LAST_PAYLOAD
from encoders.label_container import *
from log_util.event_attributes import unique_events2, unique_events
from .simple_index import simple_index


def encode_label_logs(training_log: list, test_log: list, encoding: EncodingContainer, job_type: str,
                      label: LabelContainer, additional_columns=None, balance=False, fit_encoder=False):
    """Encodes and labels test set and training set as data frames

    :param additional_columns Global trace attributes for complex and last payload encoding
    :returns training_df, test_df
    """
    event_names = unique_events2(training_log, test_log)

    #TODO change labeling type if balanced selected
    if balance:
        m_label = LabelContainer()
        threshold = compute_threshold(training_log, encoding, job_type, m_label, event_names=event_names,
                                           additional_columns=additional_columns)
        m_label = LabelContainer(type=label.type, attribute_name=label.attribute_name,
                               threshold_type=label.threshold_type, threshold=threshold)
    else:
        m_label = label

    training_df = encode_label_log(training_log, encoding, job_type, m_label, event_names=event_names,
                                   additional_columns=additional_columns)
    test_df = encode_label_log(test_log, encoding, job_type, m_label, event_names=event_names,
                               additional_columns=additional_columns)
    return training_df, test_df


def compute_threshold(run_log: list, encoding: EncodingContainer, job_type: str, label: LabelContainer, event_names=None,
                     additional_columns=None):
    if event_names is None:
        event_names = unique_events(run_log)

    encoded_log = encode_log(run_log, encoding, label, event_names, additional_columns)

    # Convert strings to number
    if label.type == ATTRIBUTE_NUMBER:
        try:
            encoded_log['label'] = encoded_log['label'].apply(lambda x: float(x))
        except :
            encoded_log['label'] = encoded_log['label'].apply(lambda x: x == 'true')


    if job_type != LABELLING:
        categorical_encode(encoded_log)
    if label.threshold_type == THRESHOLD_MEAN:
        threshold_0 = encoded_log['label'].mean()
        print('Computing proper threshold to split the two sets equally')
        threshold_ = encoded_log['label'].median()
        threshold_1 = encoded_log['label'].sort_values().iloc[int(len(encoded_log['label'])/2)]
        return threshold_


def encode_label_log(run_log: list, encoding: EncodingContainer, job_type: str, label: LabelContainer, event_names=None,
                     additional_columns=None):
    if event_names is None:
        event_names = unique_events(run_log)
        
    encoded_log = encode_log(run_log, encoding, label, event_names, additional_columns)

    # Convert strings to number
    if label.type == ATTRIBUTE_NUMBER:
        try:
            encoded_log['label'] = encoded_log['label'].apply(lambda x: float(x))
        except :
            encoded_log['label'] = encoded_log['label'].apply(lambda x: x == 'true')

    # converts string values to in
    if job_type != LABELLING:
        # Labelling has no need for this encoding
        categorical_encode(encoded_log)
    # Regression only has remaining_time or number atr as label
    if job_type == REGRESSION:
        # Remove last events as worse for prediction
        # TODO filter out 0 labels. Doing it here means runtime errors for regression
        # if label.type == REMAINING_TIME:
        #     encoded_log = encoded_log.loc[encoded_log['label'] != 0.0]
        return encoded_log
    # Post processing
    if label.type == REMAINING_TIME or label.type == ATTRIBUTE_NUMBER or label.type == DURATION:
        return label_boolean(encoded_log, label)
    return encoded_log


def encode_log(run_log: list, encoding: EncodingContainer, label: LabelContainer, event_names=None,
               additional_columns=None):
    """Encodes test set and training set as data frames

    :param additional_columns Global trace attributes for complex and last payload encoding
    :returns training_df, test_df
    """

    if encoding.prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    run_df = None
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


def label_boolean(df, label: LabelContainer):
    """Label a numeric attribute as True or False based on threshold
    This is essentially a Fast/Slow classification without string labels
    By default use mean of label value
    True if under threshold value
    """
    if df['label'].dtype == bool:
        return df
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
    #TODO this potentially generates collisions and in general is a clever solution for another problem
    # see https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8


# RANDOM TESTS
# training_log, test_log
# encoding=job['encoding']
# job_type=job['type']
# label=job['label']
# additional_columns=additional_columns
#
# from encoders.common import *
# from encoders.label_container import *
#
# event_names = unique_events2(training_log, test_log)
#
# m_label = LabelContainer()
# # threshold = compute_threshold(training_log+test_log, encoding, job_type, m_label, event_names=event_names, additional_columns=additional_columns)
# run_log = training_log+test_log
# encodings = encoding
# job_type
#
# if event_names is None:
#     event_names = unique_events(run_log)
#
# encoded_log = encode_log(run_log, encoding, m_label, event_names, additional_columns)
#
# # Convert strings to number
# if label.type == ATTRIBUTE_NUMBER:
#     encoded_log['label'] = encoded_log['label'].apply(lambda x: float(x))
#
# if job_type != LABELLING:
#     categorical_encode(encoded_log)
# if m_label.threshold_type == THRESHOLD_MEAN:
#     threshold_mean = encoded_log['label'].mean()
#     print('Computing proper threshold to split the two sets equally')
#     threshold_median = encoded_log['label'].median()
#     threshold__sortpivot = encoded_log['label'].sort_values().iloc[int(len(encoded_log['label']) / 2)]
#     threshold = threshold_median
#
#
# m_label = LabelContainer(type=label.type, attribute_name=label.attribute_name, threshold_type=label.threshold_type, threshold=threshold)
#
# training_df = encode_label_log(training_log, encoding, job_type, m_label, event_names=event_names, additional_columns=additional_columns)
# test_df = encode_label_log(test_log, encoding, job_type, m_label, event_names=event_names, additional_columns=additional_columns)
#
#
# training_df.to_csv('/Users/Brisingr/Desktop/TEMP/dataset/T+T'+train_set_fn+'.csv')
# test_df.to_csv('/Users/Brisingr/Desktop/TEMP/dataset/T+T'+test_set_fn+'.csv')
