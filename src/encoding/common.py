import hashlib

from pandas import DataFrame
from pm4py.objects.log.log import EventLog

from src.encoding.boolean_frequency import frequency, boolean
from src.encoding.complex_last_payload import complex, last_payload
from src.encoding.encoder import Encoder
from src.encoding.models import Encoding, ValueEncodings
from src.jobs.models import JobTypes, Job
from src.labelling.label_container import *
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModels
from src.utils.event_attributes import unique_events
from .simple_index import simple_index


def encode_label_logs(training_log: EventLog, test_log: EventLog, job: Job, additional_columns=None):
    training_log, cols = _encode_log(training_log, job.encoding, job.labelling, additional_columns=additional_columns,
                                     cols=None)
    # TODO pass the columns of the training log
    print('\tDataset not found in cache, building..')
    test_log, _ = _encode_log(test_log, job.encoding, job.labelling, additional_columns=additional_columns, cols=cols)

    labelling = job.labelling
    if (labelling.threshold_type in [ThresholdTypes.THRESHOLD_MEAN.value, ThresholdTypes.THRESHOLD_CUSTOM.value]) and (
        labelling.type in [LabelTypes.ATTRIBUTE_NUMBER.value, LabelTypes.DURATION.value, LabelTypes.REMAINING_TIME.value]):
        if labelling.threshold_type == ThresholdTypes.THRESHOLD_MEAN.value:
            threshold = training_log['label'].astype(float).mean()

        elif labelling.threshold_type == ThresholdTypes.THRESHOLD_CUSTOM.value:
            threshold = float(labelling.threshold)
        else:
            threshold = -1
        training_log['label'] = training_log['label'].astype(float) < threshold
        test_log['label'] = test_log['label'].astype(float) < threshold

    if job.type != JobTypes.LABELLING.value and job.encoding.value_encoding != ValueEncodings.BOOLEAN.value and \
        job.predictive_model.predictive_model != PredictiveModels.TIME_SERIES_PREDICTION.value:
        # init nominal encode
        encoder = Encoder(training_log, job.encoding)
        encoder.encode(training_log, job.encoding)
        encoder.encode(test_log, job.encoding)

    return training_log, test_log


# TODO deprecate this function
def encode_label_log(run_log: EventLog, encoding: Encoding, job_type: str, labelling: Labelling, event_names=None,
                     additional_columns=None, fit_encoder=False):
    encoded_log, _ = _encode_log(run_log, encoding, labelling, additional_columns)

    # Convert strings to number
    if labelling.type == LabelTypes.ATTRIBUTE_NUMBER.value:
        try:
            encoded_log['label'] = encoded_log['label'].apply(lambda x: float(x))
        except:
            encoded_log['label'] = encoded_log['label'].apply(lambda x: x == 'true')

    # converts string values to in
    if job_type != JobTypes.LABELLING.value:
        # Labelling has no need for this encoding
        _categorical_encode(encoded_log)
    # Regression only has remaining_time or number atr as label
    if job_type == PredictiveModels.REGRESSION.value:
        # Remove last events as worse for prediction
        # TODO filter out 0 labels. Doing it here means runtime errors for regression
        # if label.type == REMAINING_TIME:
        #     encoded_log = encoded_log.loc[encoded_log['label'] != 0.0]
        return encoded_log
    # Post processing
    if labelling.type == LabelTypes.REMAINING_TIME.value or labelling.type == LabelTypes.ATTRIBUTE_NUMBER.value or labelling.type == LabelTypes.DURATION.value:
        return _label_boolean(encoded_log, labelling)
    return encoded_log


def _encode_log(log: EventLog, encoding: Encoding, labelling: Labelling, additional_columns=None, cols=None):
    if encoding.prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    if encoding.value_encoding == ValueEncodings.SIMPLE_INDEX.value:
        run_df = simple_index(log, labelling, encoding)
    elif encoding.value_encoding == ValueEncodings.BOOLEAN.value:
        if cols is None:
            cols = unique_events(log)
        run_df = boolean(log, cols, labelling, encoding)
    elif encoding.value_encoding == ValueEncodings.FREQUENCY.value:
        if cols is None:
            cols = unique_events(log)
        run_df = frequency(log, cols, labelling, encoding)
    elif encoding.value_encoding == ValueEncodings.COMPLEX.value:
        run_df = complex(log, labelling, encoding, additional_columns)
    elif encoding.value_encoding == ValueEncodings.LAST_PAYLOAD.value:
        run_df = last_payload(log, labelling, encoding, additional_columns)
    else:
        raise ValueError("Unknown value encoding method {}".format(encoding.value_encoding))
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
