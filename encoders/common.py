from core.constants import SIMPLE_INDEX, NEXT_ACTIVITY, BOOLEAN, FREQUENCY
from encoders.boolean_frequency import frequency
from .boolean_frequency import boolean
from .log_util import unique_events2
from .simple_index import simple_index


def encode_logs(training_log: list, test_log: list, encoding_type: str, job_type: str, prefix_length=1):
    """Encodes test set and training set as data frames

    :param prefix_length only for SIMPLE_INDEX
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
    return training_df, test_df
