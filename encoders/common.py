from core.constants import SIMPLE_INDEX, NEXT_ACTIVITY, BOOLEAN, FREQUENCY, COMPLEX, LAST_PAYLOAD
from encoders.boolean_frequency import frequency
from encoders.complex_last_payload import complex, last_payload
from .boolean_frequency import boolean
from .log_util import unique_events2
from .simple_index import simple_index


def encode_logs(training_log: list, test_log: list, encoding_type: str, job_type: str, prefix_length=1,
                zero_padding=False):
    """Encodes test set and training set as data frames

    :param prefix_length only for SIMPLE_INDEX, COMPLEX, LAST_PAYLOAD
    :returns training_df, test_df
    """
    event_names = unique_events2(training_log, test_log)
    training_df = encode_log(training_log, encoding_type, job_type, prefix_length, event_names, zero_padding=zero_padding)
    test_df = encode_log(test_log, encoding_type, job_type, prefix_length, event_names, zero_padding=zero_padding)
    return training_df, test_df


def encode_log(run_log: list, encoding_type: str, job_type: str, prefix_length=1, event_names=None, add_label=True,
               zero_padding=False):
    """Encodes test set and training set as data frames

    :param prefix_length only for SIMPLE_INDEX, COMPLEX, LAST_PAYLOAD
    :param add_label True to add label to df, like remaining_time and elapsed_time
    :returns training_df, test_df
    """
    run_df = None
    if encoding_type == SIMPLE_INDEX:
        next_activity = job_type == NEXT_ACTIVITY
        run_df = simple_index(run_log, event_names, prefix_length=prefix_length, next_activity=next_activity,
                              add_label=add_label, zero_padding=zero_padding)
    elif encoding_type == BOOLEAN:
        run_df = boolean(run_log, event_names, add_label=add_label)
    elif encoding_type == FREQUENCY:
        run_df = frequency(run_log, event_names, add_label=add_label)
    elif encoding_type == COMPLEX:
        run_df = complex(run_log, event_names, prefix_length=prefix_length, add_label=add_label,
                         zero_padding=zero_padding)
    elif encoding_type == LAST_PAYLOAD:
        run_df = last_payload(run_log, event_names, prefix_length=prefix_length, add_label=add_label,
                              zero_padding=zero_padding)
    return run_df
