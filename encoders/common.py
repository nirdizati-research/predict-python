from core.constants import SIMPLE_INDEX, NEXT_ACTIVITY
from .log_util import unique_events2
from .simple_index import simple_index


def encode_logs(training_log: list, test_log: list, encoding_type: str, job_type: str, prefix_length=1):
    """Encodes test set and training set as data frames

    :returns training_df, test_df
    """
    event_names = unique_events2(training_log, test_log)
    training_df = None
    test_df = None
    if encoding_type == SIMPLE_INDEX:
        next_activity = job_type == NEXT_ACTIVITY
        training_df = simple_index(training_log, event_names, prefix_length=prefix_length, next_activity=next_activity)
        test_df = simple_index(test_log, event_names, prefix_length=prefix_length, next_activity=next_activity)
    return training_df, test_df
