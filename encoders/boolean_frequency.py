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
    events = unique_events(log)

    columns = np.append(events, list(DEFAULT_COLUMNS))
    encoded_data = []

    for trace in log:
        trace_name = classifier.get_class_identity(trace)
        event_happened = [False] * len(events)
        for event in trace:
            case_data = []
            event_name = classifier.get_class_identity(event)
            event_index = events.index(event_name)
            event_happened[event_index] = True
            case_data += event_happened

            case_data.append(trace_name)
            case_data.append(event_index)
            remaining_time = remaining_time_event(trace, event)
            case_data.append(remaining_time)
            elapsed_time = elapsed_time_event(trace, event)
            case_data.append(elapsed_time)
            encoded_data.append(case_data)

    return pd.DataFrame(columns=columns, data=encoded_data)
