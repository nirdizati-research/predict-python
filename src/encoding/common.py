import hashlib

from pandas import DataFrame

from src.encoding.boolean_frequency import frequency, boolean
from src.encoding.complex_last_payload import complex, last_payload
from src.encoding.encoding_container import EncodingContainer
from src.encoding.models import Encoding, ValueEncodings
from src.jobs.models import JobTypes
from src.labelling.label_container import *
from src.predictive_model.models import PredictiveModelTypes
from src.utils.event_attributes import unique_events
from .simple_index import simple_index


def encode_label_logs(training_log: list, test_log: list, encoding: EncodingContainer, job_type: JobTypes,
                      label: LabelContainer, additional_columns=None, split_id=None):
    """encodes and labels test set and training set as data frames

    :param training_log: 
    :param test_log: 
    :param encoding: 
    :param job_type: 
    :param label: 
    :param additional_columns: Global trace attributes for complex and last payload encoding
    :return: training_df, test_df
    """  # TODO: complete documentation
    training_log, cols = _encode_log(training_log, encoding, label, additional_columns=additional_columns,
                                     cols=None)
    # TODO pass the columns of the training log
    test_log, _ = _encode_log(test_log, encoding, label, additional_columns=additional_columns, cols=cols)

    if (label.threshold_type in [ThresholdTypes.THRESHOLD_MEAN, ThresholdTypes.THRESHOLD_CUSTOM.value]) and (
        label.type in [LabelTypes.ATTRIBUTE_NUMBER.value, LabelTypes.DURATION.value]):
        if label.threshold_type == ThresholdTypes.THRESHOLD_MEAN:
            threshold = training_log['label'].mean()
        elif label.threshold_type == ThresholdTypes.THRESHOLD_CUSTOM.value:
            threshold = label.threshold
        else:
            threshold = -1
        training_log['label'] = training_log['label'] < threshold
        test_log['label'] = test_log['label'] < threshold

    if job_type != JobTypes.LABELLING.value and encoding.method != ValueEncodings.BOOLEAN.value:
        # init nominal encode
        encoding.init_label_encoder(training_log)
        # encode data
        encoding.encode(training_log)
        encoding.encode(test_log)

    #TODO: check proper usage
    Encoding.objects.create(
        data_encoding=encoding.method, #TODO: @Hitluca check which is the proper whay to handle this
        value_encoding=encoding.generation_type,
        additional_features=label.add_remaining_time or label.add_elapsed_time or label.add_executed_events or
                            label.add_resources_used or label.add_new_traces,
        temporal_features=label.add_remaining_time or label.add_elapsed_time,
        intercase_features=label.add_executed_events or label.add_resources_used or label.add_new_traces,
        features={'features': list(training_log.columns.values)},
        prefix_len=encoding.prefix_length,
        padding=encoding.is_zero_padding()
    )

    return training_log, test_log


#TODO deprecate this function
def encode_label_log(run_log: list, encoding: EncodingContainer, job_type: str, label: LabelContainer, event_names=None,
                     additional_columns=None, fit_encoder=False):
    encoded_log, _ = _encode_log(run_log, encoding, label, additional_columns)

    # Convert strings to number
    if label.type == LabelTypes.ATTRIBUTE_NUMBER.value:
        try:
            encoded_log['label'] = encoded_log['label'].apply(lambda x: float(x))
        except:
            encoded_log['label'] = encoded_log['label'].apply(lambda x: x == 'true')

    # converts string values to in
    if job_type != JobTypes.LABELLING.value:
        # Labelling has no need for this encoding
        _categorical_encode(encoded_log)
    # Regression only has remaining_time or number atr as label
    if job_type == PredictiveModelTypes.REGRESSION.value:
        # Remove last events as worse for prediction
        # TODO filter out 0 labels. Doing it here means runtime errors for regression
        # if label.type == REMAINING_TIME:
        #     encoded_log = encoded_log.loc[encoded_log['label'] != 0.0]
        return encoded_log
    # Post processing
    if label.type == LabelTypes.REMAINING_TIME.value or label.type == LabelTypes.ATTRIBUTE_NUMBER.value or label.type == LabelTypes.DURATION.value:
        return _label_boolean(encoded_log, label)
    return encoded_log


def _encode_log(log: list, encoding: EncodingContainer, label: LabelContainer, additional_columns=None, cols=None):
    if encoding.prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    if encoding.method == ValueEncodings.SIMPLE_INDEX.value:
        run_df = simple_index(log, label, encoding)
    elif encoding.method == ValueEncodings.BOOLEAN.value:
        if cols is None:
            cols = unique_events(log)
        run_df = boolean(log, cols, label, encoding)
    elif encoding.method == ValueEncodings.FREQUENCY.value:
        if cols is None:
            cols = unique_events(log)
        run_df = frequency(log, cols, label, encoding)
    elif encoding.method == ValueEncodings.COMPLEX.value:
        run_df = complex(log, label, encoding, additional_columns)
    elif encoding.method == ValueEncodings.LAST_PAYLOAD.value:
        run_df = last_payload(log, label, encoding, additional_columns)
    else:
        raise ValueError("Unknown encoding method {}".format(encoding.method))
    return run_df, cols


def _label_boolean(df: DataFrame, label: LabelContainer) -> DataFrame:
    """Label a numeric attribute as True or False based on threshold

    This is essentially a Fast/Slow classification without string labels. By default use mean of label value True if 7
    under threshold value
    :param df:
    :param label:
    :return:
    """
    if df['label'].dtype == bool:
        return df
    if label.threshold_type == ThresholdTypes.THRESHOLD_MEAN.value:
        threshold = df['label'].mean()
    else:
        threshold = float(label.threshold)
    df['label'] = df['label'] < threshold
    return df


def _categorical_encode(df: DataFrame) -> DataFrame:
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
    # TODO this potentially generates collisions and in general is a clever solution for another problem
    # see https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8
