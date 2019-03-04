import hashlib
import pickle

from pandas import DataFrame

from src.cache.models import LabelledLogs, LoadedLog
from src.jobs.models import Job
from src.split.models import Split


def get_digested(candidate_path: str) -> str:
    return hashlib.sha256(candidate_path.encode('utf-8')).hexdigest()


def load_from_cache(path: str, prefix: str = ''):
    with open(prefix + get_digested(path) + '.pickle', 'rb') as f:
        obj = pickle.load(f)
    return obj


def dump_to_cache(path: str, obj, prefix: str = ''):
    with open(prefix + get_digested(path) + '.pickle', "wb") as f:
        pickle.dump(obj, f)


def put_loaded_logs(split: Split, train_df, test_df, additional_columns):
    [dump_to_cache(path, data) for (path, data) in [
        (split.train_log.path, train_df),
        (split.test_log.path, test_df),
        (split.additional_columns.path, additional_columns)
    ]]
    LoadedLog.objects.create(train_log=split.train_log.path,
                             test_log=split.test_log.path,
                             additional_columns=split.additional_columns.path)


def put_labelled_logs(job: Job, train_df, test_df):
    [dump_to_cache(path, data) for (path, data) in [
        (job.split.train_log.path, train_df),
        (job.split.test_log.path, test_df),
    ]]
    LabelledLogs.objects.create(split=job.split,
                                encoding=job.encoding,
                                labelling=job.labelling)


def get_loaded_logs(split: Split) -> (DataFrame, DataFrame):
    print('\t\tFound Dataset in cache, loading..')
    cache = LoadedLog.objects.filter(train_log=split.train_log.path,
                                     test_log=split.test_log.path)[0]
    return (
        load_from_cache(cache.train_log.name),
        load_from_cache(cache.test_log.name)
    )


def get_labelled_logs(job: Job) -> (DataFrame, DataFrame):
    print('\t\tFound Dataset in cache, loading..')
    cache = LabelledLogs.objects.filter(split=job.split,
                                        encoding=job.encoding,
                                        labelling=job.labelling)[0]
    return (
        load_from_cache(cache.train_log.name),
        load_from_cache(cache.test_log.name)
    )
