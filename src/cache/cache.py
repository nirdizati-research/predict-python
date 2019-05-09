import hashlib
import pickle

from pandas import DataFrame

from src.cache.models import LabelledLog, LoadedLog
from src.jobs.models import Job
from src.split.models import Split


def get_digested(candidate_path: str) -> str:
    return hashlib.sha256(candidate_path.encode('utf-8')).hexdigest()


def load_from_cache(path: str, prefix: str = ''):
    if path is not None: #TODO: what if the file is not there?
        with open(prefix + get_digested(path) + '.pickle', 'rb') as f:
            return pickle.load(f)


def dump_to_cache(path: str, obj, prefix: str = ''):
    if path is not None:
        with open(prefix + get_digested(path) + '.pickle', "wb") as f:
            pickle.dump(obj, f)


def put_loaded_logs(split: Split, train_df, test_df, additional_columns):
    [dump_to_cache(path, data, 'cache/loaded_log_cache/') for (path, data) in [
        (split.train_log.name, train_df),
        (split.test_log.name, test_df),
        (split.additional_columns, additional_columns)
    ]]
    LoadedLog.objects.create(train_log_path=split.train_log.name,
                             test_log_path=split.test_log.name,
                             additional_columns_path=split.additional_columns,
                             split=split)


def put_labelled_logs(job: Job, train_df, test_df):
    [dump_to_cache(path, data, 'cache/labeled_log_cache/') for (path, data) in [
        (job.split.train_log.name, train_df),
        (job.split.test_log.name, test_df)
    ]]
    LabelledLog.objects.create(split=job.split,
                               encoding=job.encoding,
                               labelling=job.labelling,
                               train_log_path=job.split.train_log.name,
                               test_log_path=job.split.test_log.name)


def get_loaded_logs(split: Split) -> (DataFrame, DataFrame, DataFrame):
    print('\t\tFound pre-loaded Dataset in cache, loading..')
    cache = LoadedLog.objects.filter(split=split)[0]
    return (
        load_from_cache(path=cache.train_log_path, prefix='cache/loaded_log_cache/'),
        load_from_cache(path=cache.test_log_path, prefix='cache/loaded_log_cache/'),
        load_from_cache(path=cache.additional_columns_path, prefix='cache/loaded_log_cache/')
    )


def get_labelled_logs(job: Job) -> (DataFrame, DataFrame):
    print('\t\tFound pre-labeled Dataset in cache, loading..')
    cache = LabelledLog.objects.filter(split=job.split,
                                       encoding=job.encoding,
                                       labelling=job.labelling)[0]
    return (
        load_from_cache(path=cache.train_log_path, prefix='cache/labeled_log_cache/'),
        load_from_cache(path=cache.test_log_path, prefix='cache/labeled_log_cache/')
    )
