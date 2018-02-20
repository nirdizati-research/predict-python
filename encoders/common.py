from core.constants import SIMPLE_INDEX, NEXT_ACTIVITY, BOOLEAN, FREQUENCY, COMPLEX, LAST_PAYLOAD
from encoders.boolean_frequency import frequency
from encoders.complex_last_payload import complex, last_payload
from .boolean_frequency import boolean
from .log_util import unique_events2, unique_events
from .simple_index import simple_index
from cProfile import run


def encode_logs(training_log: list, test_log: list, encoding_type: str, job_type: str, prefix_length=1):
    """Encodes test set and training set as data frames

    :param prefix_length only for SIMPLE_INDEX, COMPLEX, LAST_PAYLOAD
    :returns training_df, test_df
    """
    event_names = unique_events2(training_log, test_log)
    training_df = None
    test_df = None
    if encoding_type == SIMPLE_INDEX:
        next_activity = job_type == NEXT_ACTIVITY
        training_df = simple_index(training_log, event_names, prefix_length=prefix_length, next_activity=next_activity)
        test_df = simple_index(test_log, event_names, prefix_length=prefix_length, next_activity=next_activity)
    elif encoding_type == BOOLEAN:
        training_df = boolean(training_log, event_names)
        test_df = boolean(test_log, event_names)
    elif encoding_type == FREQUENCY:
        training_df = frequency(training_log, event_names)
        test_df = frequency(test_log, event_names)
    elif encoding_type == COMPLEX:
        training_df = complex(training_log, event_names)
        test_df = complex(test_log, event_names)
    elif encoding_type == LAST_PAYLOAD:
        training_df = last_payload(training_log, event_names)
        test_df = last_payload(test_log, event_names)
    return training_df, test_df

def encode_training_logs(training_log: list, encoding_type: str, job_type: str, prefix_length=1):
    """Encodes test set and training set as data frames

    :param prefix_length only for SIMPLE_INDEX, COMPLEX, LAST_PAYLOAD
    :returns training_df, test_df
    """
    event_names = unique_events(training_log)
    training_df = None
    if encoding_type == SIMPLE_INDEX:
        next_activity = job_type == NEXT_ACTIVITY
        training_df = simple_index(training_log, event_names, prefix_length=prefix_length, next_activity=next_activity)
    elif encoding_type == BOOLEAN:
        training_df = boolean(training_log, event_names)
    elif encoding_type == FREQUENCY:
        training_df = frequency(training_log, event_names)
    elif encoding_type == COMPLEX:
        training_df = complex(training_log, event_names)
    elif encoding_type == LAST_PAYLOAD:
        training_df = last_payload(training_log, event_names)
    return training_df

def encode_run_logs(run_log: list, encoding_type: str, job_type: str, prefix_length=1):
    """Encodes test set and training set as data frames

    :param prefix_length only for SIMPLE_INDEX, COMPLEX, LAST_PAYLOAD
    :returns training_df, test_df
    """
    event_names = unique_events(run_log)
    run_df = None
    if encoding_type == SIMPLE_INDEX:
        next_activity = job_type == NEXT_ACTIVITY
        run_df = simple_index(run_log, event_names, prefix_length=prefix_length, next_activity=next_activity, run=True)
    elif encoding_type == BOOLEAN:
        run_df = boolean(run_log, event_names)
    elif encoding_type == FREQUENCY:
        run_df = frequency(run_log, event_names)
    elif encoding_type == COMPLEX:
        run_df = complex(run_log, event_names, prefix_length, run=True)
    elif encoding_type == LAST_PAYLOAD:
        run_df = last_payload(run_log, event_names, prefix_length, run=True)
    return run_df
