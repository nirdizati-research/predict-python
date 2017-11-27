from core.constants import SIMPLE_INDEX, NEXT_ACTIVITY, BOOLEAN, FREQUENCY, COMPLEX, LAST_PAYLOAD
from encoders.boolean_frequency import frequency
from encoders.complex_last_payload import complex, last_payload
from .boolean_frequency import boolean
from .log_util import unique_events2
from .simple_index import simple_index


def encode_logs(training_log: list, test_log: list, encoding_type: str, job_type: str, prefix_length=1):
    """Encodes test set and training set as data frames

    :param prefix_length only for SIMPLE_INDEX, COMPLEX, LAST_PAYLOAD
    :returns training_df, test_df
    """
    event_names = unique_events2(training_log, test_log)
    training_df = None
    test_df = None
    if encoding_type is SIMPLE_INDEX:
        next_activity = job_type == NEXT_ACTIVITY
        training_df = simple_index(training_log, event_names, prefix_length=prefix_length, next_activity=next_activity)
        test_df = simple_index(test_log, event_names, prefix_length=prefix_length, next_activity=next_activity)
    elif encoding_type is BOOLEAN:
        training_df = boolean(training_log, event_names)
        test_df = boolean(test_log, event_names)
    elif encoding_type is FREQUENCY:
        training_df = frequency(training_log, event_names)
        test_df = frequency(test_log, event_names)
    elif encoding_type is COMPLEX:
        training_df = complex(training_log, event_names)
        test_df = complex(test_log, event_names)
    elif encoding_type is LAST_PAYLOAD:
        training_df = last_payload(training_log, event_names)
        test_df = last_payload(test_log, event_names)
    return training_df, test_df
