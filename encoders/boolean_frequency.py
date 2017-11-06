import numpy as np
import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from encoders.log_util import DEFAULT_COLUMNS, unique_events, remaining_time_event, \
    elapsed_time_event

classifier = XEventAttributeClassifier("Trace name", ["concept:name"])


def boolean(data):
    return encode_boolean_frequency(data, 'boolean')


def frequency(data):
    return encode_boolean_frequency(data, 'frequency')


def encode_boolean_frequency(log: list, encoding='boolean'):
    """Internal method for both boolean and frequency. Only dif is __append_item"""
    event_names = unique_events(log)

    columns = np.append(event_names, list(DEFAULT_COLUMNS))
    encoded_data = []

    for trace in log:
        trace_name = classifier.get_class_identity(trace)
        # starts with all False, changes to event
        event_happened = [False] * len(event_names)
        for event_index, event in enumerate(trace):
            case_data = []
            update_event_happened(event, event_names, event_happened)
            case_data += event_happened

            case_data.append(trace_name)
            # Start counting at 1
            case_data.append(event_index + 1)
            remaining_time = remaining_time_event(trace, event)
            case_data.append(remaining_time)
            elapsed_time = elapsed_time_event(trace, event)
            case_data.append(elapsed_time)
            encoded_data.append(case_data)

    return pd.DataFrame(columns=columns, data=encoded_data)


def update_event_happened(event, event_names: list, event_happened: list):
    """Updates the event_happened list at event index"""
    event_name = classifier.get_class_identity(event)
    event_index = event_names.index(event_name)
    event_happened[event_index] = True
