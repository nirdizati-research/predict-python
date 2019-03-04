import functools
from typing import Union

from pm4py.objects.log.log import EventLog
from sklearn.model_selection import train_test_split

from src.split.models import Split, SplitTypes, SplittingMethods
from src.utils.event_attributes import get_additional_columns
from src.utils.file_service import get_log


def prepare_logs(split: Split):
    """Returns training_log and test_log"""
    if split.type == SplitTypes.SPLIT_SINGLE.value:
        additional_columns = get_additional_columns(get_log(split.original_log))
        training_log, test_log = _split_single_log(split)
        print("\t\tLoaded single log from {}".format(split.original_log.path))
    else:
        # Have to use sklearn to convert some internal data types
        training_log = get_log(split.train_log)
        additional_columns = get_additional_columns(training_log)
        training_log, _ = train_test_split(training_log, test_size=0, shuffle=False)
        test_log, _ = train_test_split(get_log(split.test_log), test_size=0, shuffle=False)
        print("\t\tLoaded double logs from {} and {}.".format(split.train_log.path, split.test_log.path))
    if len(training_log) == 0:
        raise TypeError("Training log is empty. Create a new Split with better parameters")
    return training_log, test_log, additional_columns


def _split_single_log(split: Split):
    log = get_log(split.original_log)
    print("\t\tExecute single split ID {}, split_type {}, test_size {}".format(split.id, split.type, split.test_size))
    if split.splitting_method == SplittingMethods.SPLIT_TEMPORAL.value:
        return _temporal_split(log, split.test_size)
    elif split.splitting_method == SplittingMethods.SPLIT_STRICT_TEMPORAL.value:
        return _temporal_split_strict(log, split.test_size)
    elif split.splitting_method == SplittingMethods.SPLIT_SEQUENTIAL.value:
        return _split_log(log, split.test_size, shuffle=False)
    elif split.splitting_method == SplittingMethods.SPLIT_RANDOM.value:
        return _split_log(log, split.test_size, random_state=None)
    else:
        raise ValueError("Unknown splitting method", split.splitting_method)


def _temporal_split(log: EventLog, test_size: float):
    """sort log by first event timestamp to enforce temporal order"""
    log = sorted(log, key=functools.cmp_to_key(_compare_trace_starts))
    training_log, test_log = train_test_split(log, test_size=test_size, shuffle=False)
    return training_log, test_log


def _temporal_split_strict(log: EventLog, test_size: float):
    """Includes only training traces where it's last event ends before the first in test trace"""
    training_log, test_log = _temporal_split(log, test_size)
    test_first_time = _trace_event_time(test_log[0])
    training_log = filter(lambda x: _trace_event_time(x, event_index=-1) < test_first_time, training_log)
    return list(training_log), test_log


def _split_log(log: EventLog, test_size: float, random_state: Union[int, None] = 4, shuffle=True):
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
