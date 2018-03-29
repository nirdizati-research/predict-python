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
    event_names=unique_events2(training_log,test_log)
    training_df=encode_log(training_log, encoding_type, job_type, prefix_length,event_names)
    test_df=encode_log(test_log, encoding_type, job_type, prefix_length,event_names)
    return training_df, test_df

def encode_training_logs(training_log: list, test_log:list, encoding_type: str, job_type: str, prefix_length=0):
    """Encodes test set and training set as data frames

    :param prefix_length only for SIMPLE_INDEX, COMPLEX, LAST_PAYLOAD
    :returns training_df, test_df
    """
    if not prefix_length == 0:
        return encode_log(training_log, encoding_type, job_type, prefix_length), encode_log(test_log, encoding_type, job_type, prefix_length)
    else:
        event_names, prefix_length = unique_events(training_log)
        training_df = dict()
        training_df['prefix_length']=prefix_length
        
        if encoding_type == BOOLEAN or encoding_type == FREQUENCY:
            event_names=unique_events2(training_log,test_log)
            training_df[prefix_length]=encode_log(training_log, encoding_type, job_type, prefix_length, event_names)
            return training_df
        
        for i in range(prefix_length):
            if i < 1:
                pass
            else:
                training_df[i]=encode_log(training_log, encoding_type, job_type, i, event_names)
        
        return training_df

def encode_log(run_log: list, encoding_type: str, job_type: str, prefix_length=0, event_names=None):
    """Encodes test set and training set as data frames

    :param prefix_length only for SIMPLE_INDEX, COMPLEX, LAST_PAYLOAD
    :returns training_df, test_df
    """
    run=False
    if event_names is None:
        if prefix_length == 0:
            event_names, prefix_length = unique_events(run_log)
            run=True
        else:
            event_names, _ = unique_events(run_log)
            run=False
    run_df =None
    if encoding_type == SIMPLE_INDEX:
        next_activity = job_type == NEXT_ACTIVITY
        run_df = simple_index(run_log, event_names, prefix_length=prefix_length, next_activity=next_activity, run=run)
    elif encoding_type == BOOLEAN:
        run_df = boolean(run_log, event_names)
    elif encoding_type == FREQUENCY:
        run_df = frequency(run_log, event_names)
    elif encoding_type == COMPLEX:
        run_df = complex(run_log, event_names, prefix_length=prefix_length, run=run)
    elif encoding_type == LAST_PAYLOAD:
        run_df = last_payload(run_log, event_names, prefix_length=prefix_length, run=run)
    if run:
        return run_df, prefix_length
    else:
        return run_df
