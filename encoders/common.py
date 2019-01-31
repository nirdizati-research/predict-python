import hashlib

from core.constants import LABELLING, REGRESSION
from encoders.boolean_frequency import frequency, boolean
from encoders.complex_last_payload import complex, last_payload
from encoders.encoding_container import EncodingContainer, SIMPLE_INDEX, BOOLEAN, FREQUENCY, COMPLEX, LAST_PAYLOAD
from encoders.label_container import *
from utils.event_attributes import unique_events2, unique_events
from .simple_index import simple_index


def encode_label_logs_new(training_log: list, test_log: list, encoding: EncodingContainer, job_type: str,
                      label: LabelContainer, additional_columns=None, balance=False):
    training_log = encode_log(training_log, encoding, label, additional_columns)

    # TODO ATTRIBUTE_NUMBER not anymore supported

    # TODO Extremely bad workaround to label dataset in a more balanced way
    if balance and label.threshold_type == THRESHOLD_MEAN:
        print('Computing proper threshold to split the two sets equally')
        threshold = training_log['label'].median()
        label = LabelContainer(type=label.type, attribute_name=label.attribute_name,
                               threshold_type=label.threshold_type, threshold=threshold)
    #TODO pass the columns of the training log
    test_log = encode_log(test_log, encoding, label, additional_columns)

    if job_type != LABELLING:
        #init nominal encode
        encoding.init_label_encoder(training_log)
        #encode data
        encoding.encode(training_log)
        encoding.encode(test_log)

    return training_log, test_log


def encode_label_logs(training_log: list, test_log: list, encoding: EncodingContainer, job_type: str,
                      label: LabelContainer, additional_columns=None):
    """Encodes and labels test set and training set as data frames

    :param additional_columns: Global trace attributes for complex and last payload encoding
    :returns training_df, test_df
    """  # TODO: complete documentation
    training_log = encode_log(training_log, encoding, label, additional_columns=additional_columns)

    # TODO pass the columns of the training log
    test_log = encode_log(test_log, encoding, label, additional_columns=additional_columns)

    if (label.threshold_type == THRESHOLD_MEAN or
        label.threshold_type == THRESHOLD_CUSTOM) and (label.type == REMAINING_TIME or
                                                       label.type == ATTRIBUTE_NUMBER or
                                                       label.type == DURATION):
        if label.threshold_type == THRESHOLD_MEAN:
            threshold = training_log['label'].mean()
        elif label.threshold_type == THRESHOLD_CUSTOM:
            threshold = label.threshold
        training_log['label'] = training_log['label'] < threshold
        test_log['label'] = test_log['label'] < threshold

    if job_type != LABELLING:
        # init nominal encode
        encoding.init_label_encoder(training_log)
        # encode data
        encoding.encode(training_log)
        encoding.encode(test_log)

    return training_log, test_log


def encode_label_log(run_log: list, encoding: EncodingContainer, job_type: str, label: LabelContainer, event_names=None,
                     additional_columns=None, fit_encoder=False):
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


def encode_log(log: list, encoding: EncodingContainer, label: LabelContainer,
               additional_columns=None):
    """Encodes test set and training set as data frames

    :param additional_columns: Global trace attributes for complex and last payload encoding
    :returns training_df, test_df
    """  # TODO: complete documentation

    if encoding.prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    if encoding.method == SIMPLE_INDEX:
        run_df = simple_index(log, label, encoding)
    elif encoding.method == BOOLEAN:
        event_names = unique_events(log)
        run_df = boolean(log, event_names, label, encoding)
    elif encoding.method == FREQUENCY:
        event_names = unique_events(log)
        run_df = frequency(log, event_names, label, encoding)
    elif encoding.method == COMPLEX:
        run_df = complex(log, label, encoding, additional_columns)
    elif encoding.method == LAST_PAYLOAD:
        run_df = last_payload(log, label, encoding, additional_columns)
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
