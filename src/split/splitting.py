import functools

from sklearn.model_selection import train_test_split

from src.utils.event_attributes import get_additional_columns
from src.utils.file_service import get_log

SPLIT_SEQUENTIAL = 'split_sequential'
SPLIT_TEMPORAL = 'split_temporal'
SPLIT_RANDOM = 'split_random'
SPLIT_STRICT_TEMPORAL = 'split_strict_temporal'


def prepare_logs(split: dict):
    """Returns training_log and test_log"""
    if split['type'] == 'single':
        log = get_log(split['original_log_path'])
        additional_columns = get_additional_columns(log)
        training_log, test_log = _split_single_log(split, log)
        print("Loaded single log from {}".format(split['original_log_path']))
    else:
        # Have to use sklearn to convert some internal data types
        training_log = get_log(split['training_log_path'])
        additional_columns = get_additional_columns(training_log)
        training_log, _ = train_test_split(training_log, test_size=0, shuffle=False)
        test_log, _ = train_test_split(get_log(split['test_log_path']), test_size=0, shuffle=False)
        print("Loaded double logs from {} and {}.".format(split['training_log_path'], split['test_log_path']))
    if len(training_log) == 0:
        raise TypeError("Training log is empty. Create a new Split with better parameters")
    return training_log, test_log, additional_columns


def _split_single_log(split: dict, log: list):
    test_size = split['config'].get('test_size', 0.2)
    if test_size <= 0 or test_size >= 1:
        print("Using out of bound split test_size {}. Reverting to default 0.2.".format(test_size))
        test_size = 0.2
    split_type = split['config'].get('split_type', SPLIT_SEQUENTIAL)
    print("Execute single split ID {}, split_type {}, test_size {}".format(split['id'], split_type, test_size))
    if split_type == SPLIT_TEMPORAL:
        return _temporal_split(log, test_size)
    elif split_type == SPLIT_STRICT_TEMPORAL:
        return _temporal_split_strict(log, test_size)
    elif split_type == SPLIT_SEQUENTIAL:
        return _split_log(log, test_size=test_size, shuffle=False)
    elif split_type == SPLIT_RANDOM:
        return _split_log(log, test_size=test_size, random_state=None)
    else:
        raise ValueError("Unknown split type", split_type)


def _temporal_split(log: list, test_size: float):
    """sort log by first event timestamp to enforce temporal order"""
    log = sorted(log, key=functools.cmp_to_key(_compare_trace_starts))
    training_log, test_log = train_test_split(log, test_size=test_size, shuffle=False)
    return training_log, test_log


def _temporal_split_strict(log: list, test_size: float):
    """Includes only training traces where it's last event ends before the first in test trace"""
    training_log, test_log = _temporal_split(log, test_size)
    test_first_time = _trace_event_time(test_log[0])
    training_log = filter(lambda x: _trace_event_time(x, event_index=-1) < test_first_time, training_log)
    return list(training_log), test_log


def _split_log(log: list, test_size=0.2, random_state=4, shuffle=True):
    training_log, test_log = train_test_split(log, test_size=test_size, random_state=random_state, shuffle=shuffle)
    return training_log, test_log


def _compare_trace_starts(item1, item2):
    first = _trace_event_time(item1)
    second = _trace_event_time(item2)
    if first < second:
        return -1
    elif first > second:
        return 1
    else:
        return 0


def _trace_event_time(trace, event_index=0):
    """Event time as Date. By default first event."""
    return trace[event_index]['time:timestamp']
