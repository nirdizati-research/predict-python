import hashlib
import logging

from pandas import DataFrame
from pm4py.objects.log.log import EventLog

from src.encoding.boolean_frequency import frequency, boolean
from src.encoding.complex_last_payload import complex, last_payload
# from src.encoding.declare.sequence import sequences
from src.encoding.declare.declare import declare_encoding
from src.encoding.encoder import Encoder
from src.encoding.models import Encoding, ValueEncodings
from src.jobs.models import JobTypes, Job
from src.labelling.label_container import *
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModels
from src.split.splitting import get_train_test_log
from src.utils.event_attributes import unique_events
from .simple_index import simple_index
from ..cache.cache import get_labelled_logs, get_loaded_logs, put_loaded_logs, put_labelled_logs
from ..cache.models import LabelledLog, LoadedLog
from ..logs.log_service import create_log
from ..split.models import SplitTypes, Split
from ..utils.django_orm import duplicate_orm_row

logger = logging.getLogger(__name__)


def encode_label_logs(training_log: EventLog, test_log: EventLog, job: Job, additional_columns=None, encode=True):
    logger.info('\tDataset not found in cache, building..')
    training_log, cols = _eventlog_to_dataframe(training_log, job.encoding, job.labelling, additional_columns=additional_columns, cols=None)
    test_log, _ = _eventlog_to_dataframe(test_log, job.encoding, job.labelling, additional_columns=additional_columns, cols=cols)

    labelling = job.labelling
    if (labelling.threshold_type in [ThresholdTypes.THRESHOLD_MEAN.value, ThresholdTypes.THRESHOLD_CUSTOM.value]) and (
        labelling.type in [LabelTypes.ATTRIBUTE_NUMBER.value, LabelTypes.DURATION.value,
                           LabelTypes.REMAINING_TIME.value]):
        if labelling.threshold_type == ThresholdTypes.THRESHOLD_MEAN.value:
            threshold = training_log['label'].astype(float).mean()

        elif labelling.threshold_type == ThresholdTypes.THRESHOLD_CUSTOM.value:
            threshold = float(labelling.threshold)
        else:
            threshold = -1
        training_log['label'] = training_log['label'].astype(float) < threshold
        test_log['label'] = test_log['label'].astype(float) < threshold
    elif (labelling.threshold_type == ThresholdTypes.NONE.value) and (
          labelling.type in [LabelTypes.ATTRIBUTE_NUMBER.value, LabelTypes.DURATION.value,
                             LabelTypes.REMAINING_TIME.value]):
        mask = training_log.applymap(type) != bool
        d = {True: 'TRUE', False: 'FALSE'}

        df = training_log.where(mask, training_log.replace(d))
        training_log['label'] = training_log['label'].astype(float)
        test_log['label'] = test_log['label'].astype(float)

    if encode:
        _data_encoder_encoder(job, training_log, test_log)

    return training_log, test_log


def _eventlog_to_dataframe(log: EventLog, encoding: Encoding, labelling: Labelling, additional_columns=None, cols=None):
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
    # elif encoding.value_encoding == ValueEncodings.SEQUENCES.value: #TODO JONAS
    #     run_df = sequences(log, labelling, encoding, additional_columns)
    elif encoding.value_encoding == ValueEncodings.DECLARE.value:
        run_df = declare_encoding(log, labelling, encoding, additional_columns, cols=cols)
        if cols is None:
            cols = list(run_df.columns)
    else:
        raise ValueError("Unknown value encoding method {}".format(encoding.value_encoding))
    return run_df, cols


def _data_encoder_encoder(job: Job, training_log, test_log) -> Encoder:
    if job.type != JobTypes.LABELLING.value and \
       job.encoding.value_encoding != ValueEncodings.BOOLEAN.value and \
       job.predictive_model.predictive_model != PredictiveModels.TIME_SERIES_PREDICTION.value:
        if job.incremental_train is not None:
            encoder = retrieve_proper_encoder(job.incremental_train)
        else:
            if job.predictive_model.predictive_model != PredictiveModels.TIME_SERIES_PREDICTION.value and \
               job.predictive_model.predictive_model != PredictiveModels.REGRESSION.value:
                encoder = Encoder(training_log, job.encoding)
            elif job.predictive_model.predictive_model == PredictiveModels.REGRESSION.value:
                encoder = Encoder(training_log.drop('label', axis=1), job.encoding)

        encoder.encode(training_log, job.encoding)
        encoder.encode(test_log, job.encoding)

        return encoder


def data_encoder_decoder(job: Job, training_log, test_log) -> None:
    encoder = retrieve_proper_encoder(job)
    encoder.decode(training_log, job.encoding), encoder.decode(test_log, job.encoding)


def retrieve_proper_encoder(job: Job) -> Encoder:
    if job.incremental_train is not None:
        return retrieve_proper_encoder(job.incremental_train)
    else:
        training_log, test_log, additional_columns = get_train_test_log(job.split)
        training_df, _ = encode_label_logs(training_log, test_log, job, additional_columns=additional_columns,
                                           encode=False)
    return Encoder(training_df, job.encoding)


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


# def _categorical_encode(df: DataFrame) -> DataFrame:
#     """Encodes every column except trace_id and label as int
#
#     Encoders module puts event name in cell, which can't be used by machine learning methods directly.
#     """
#     for column in df.columns:
#         if column == 'trace_id':
#             continue
#         elif df[column].dtype == type(str):
#             df[column] = df[column].map(lambda s: _convert(s))
#     return df
#
#
# def _convert(s):
#     if isinstance(s, float) or isinstance(s, int):
#         return s
#     if s is None:
#         # Next activity resources
#         s = '0'
#     # TODO this potentially generates collisions and in general is a clever solution for another problem
#     # see https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
#     return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8


def get_encoded_logs(job: Job, use_cache: bool = True) -> (DataFrame, DataFrame):
    """returns the encoded logs

    returns the training and test DataFrames encoded using the given job configuration, loading from cache if possible
    :param job: job configuration
    :param use_cache: load or not saved datasets from cache
    :return: training and testing DataFrame

    """
    logger.info('\tGetting Dataset')

    if use_cache and \
        (job.predictive_model is not None and
         job.predictive_model.predictive_model != PredictiveModels.TIME_SERIES_PREDICTION.value):

        if LabelledLog.objects.filter(split=job.split,
                                      encoding=job.encoding,
                                      labelling=job.labelling).exists():
            try:
                training_df, test_df = get_labelled_logs(job)
            except FileNotFoundError: #cache invalidation
                LabelledLog.objects.filter(split=job.split,
                                           encoding=job.encoding,
                                           labelling=job.labelling).delete()
                logger.info('\t\tError pre-labeled cache invalidated!')
                return get_encoded_logs(job, use_cache)
        else:
            if job.split.train_log is not None and \
               job.split.test_log is not None and \
               LoadedLog.objects.filter(split=job.split).exists():
                try:
                    training_log, test_log, additional_columns = get_loaded_logs(job.split)
                except FileNotFoundError:  # cache invalidation
                    LoadedLog.objects.filter(split=job.split).delete()
                    logger.info('\t\tError pre-loaded cache invalidated!')
                    return get_encoded_logs(job, use_cache)
            else:
                training_log, test_log, additional_columns = get_train_test_log(job.split)
                if job.split.type == SplitTypes.SPLIT_SINGLE.value:
                    search_for_already_existing_split = Split.objects.filter(
                        type=SplitTypes.SPLIT_DOUBLE.value,
                        original_log=job.split.original_log,
                        test_size=job.split.test_size,
                        splitting_method=job.split.splitting_method
                    )
                    if len(search_for_already_existing_split) >= 1:
                        job.split = search_for_already_existing_split[0]
                        job.split.save()
                        job.save()
                        return get_encoded_logs(job, use_cache=use_cache)
                    else:
                        # job.split = duplicate_orm_row(Split.objects.filter(pk=job.split.pk)[0])  #todo: replace with simple CREATE
                        job.split = Split.objects.create(
                            type=job.split.type,
                            original_log=job.split.original_log,
                            test_size=job.split.test_size,
                            splitting_method=job.split.splitting_method,
                            train_log=job.split.train_log,
                            test_log=job.split.test_log,
                            additional_columns=job.split.additional_columns
                        ) #todo: futurebug if object changes
                        job.split.type = SplitTypes.SPLIT_DOUBLE.value
                        train_name = 'SPLITTED_' + job.split.original_log.name.split('.')[0] + '_0-' + str(int(100 - (job.split.test_size * 100)))
                        job.split.train_log = create_log(
                            EventLog(training_log),
                            train_name + '.xes'
                        )
                        test_name = 'SPLITTED_' + job.split.original_log.name.split('.')[0] + '_' + str(int(100 - (job.split.test_size * 100))) + '-100'
                        job.split.test_log = create_log(
                            EventLog(test_log),
                            test_name + '.xes'
                        )
                        job.split.additional_columns = str(train_name + test_name)  # TODO: find better naming policy
                        job.split.save()

                put_loaded_logs(job.split, training_log, test_log, additional_columns)

            training_df, test_df = encode_label_logs(
                training_log,
                test_log,
                job,
                additional_columns=additional_columns)
            put_labelled_logs(job, training_df, test_df)
    else:
        training_log, test_log, additional_columns = get_train_test_log(job.split)
        training_df, test_df = encode_label_logs(training_log, test_log, job, additional_columns=additional_columns)
    return training_df, test_df
