import functools

from dateutil.parser import *
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier
from sklearn.model_selection import train_test_split

from logs.file_service import get_logs

SPLIT_SEQUENTIAL = 'split_sequential'
SPLIT_TEMPORAL = 'split_temporal'
SPLIT_RANDOM = 'split_random'
SPLIT_STRICT_TEMPORAL = 'split_strict_temporal'

TIMESTAMP_CLASSIFIER = XEventAttributeClassifier("Timestamp", ["time:timestamp"])


def prepare_logs(split: dict):
    """Returns training_log and test_log"""
    if split['type'] == 'single':
        log = get_logs(split['original_log_path'])[0]
        training_log, test_log = _split_single_log(split, log)
        print("Loaded single log from {}".format(split['original_log_path']))
    else:
        # Have to use sklearn to convert some internal data types
        training_log, _ = train_test_split(get_logs(split['training_log_path'])[0], test_size=0)
        test_log, _ = train_test_split(get_logs(split['test_log_path'])[0], test_size=0)
        print("Loaded double logs from {} and {}.".format(split['training_log_path'], split['test_log_path']))
    return training_log, test_log


def _split_single_log(split: dict, log: list):
    test_size = split['config'].get('test_size', 0.2)
    if test_size <= 0 or test_size >= 1:
        print("Using out of bound split test_size {}. Reverting to default 0.2.".format(test_size))
        test_size = 0.2
    split_type = split['config'].get('split_type', SPLIT_SEQUENTIAL)
    if split_type == SPLIT_TEMPORAL:
        return _temporal_split(log, test_size)
    elif split_type == SPLIT_SEQUENTIAL:
        return _split_log(log, test_size=test_size, shuffle=False)
    elif split_type == SPLIT_RANDOM:
        return _split_log(log, test_size=test_size, random_state=None)
    else:
        raise TypeError("Unknown split type", split_type)


def _temporal_split(log: list, test_size: float):
    # sort log by first event timestamp
    log = sorted(log, key=functools.cmp_to_key(_compare_trace_starts))
    training_log, test_log = train_test_split(log, test_size=test_size, shuffle=False)
    return training_log, test_log


def _split_log(log: list, test_size=0.2, random_state=4, shuffle=True):
    training_log, test_log = train_test_split(log, test_size=test_size, random_state=random_state, shuffle=shuffle)
    return training_log, test_log


def _compare_trace_starts(item1, item2):
    first = _trace_first_event_time(item1)
    second = _trace_first_event_time(item2)
    if first < second:
        return -1
    elif first > second:
        return 1
    else:
        return 0


def _trace_first_event_time(trace):
    """First event time in milliseconds"""
    first_time = TIMESTAMP_CLASSIFIER.get_class_identity(trace[0])
    return parse(first_time)
