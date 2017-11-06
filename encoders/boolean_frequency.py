import numpy as np
import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from encoders.log_util import unique_events, DEFAULT_COLUMNS, remaining_time, elapsed_time

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])


def boolean(data):
    return encode_boolean_frequency(data, boolean=True)


def frequency(data):
    return encode_boolean_frequency(data, boolean=False)


def encode_boolean_frequency(log: list, boolean=True):
    """Encodes the log by boolean or frequency

    :return pandas dataframe
    """
    event_names = unique_events(log)

    columns = np.append(event_names, list(DEFAULT_COLUMNS))
    encoded_data = []

    for trace in log:
        trace_name = CLASSIFIER.get_class_identity(trace)
        # starts with all False, changes to event
        event_happened = create_event_happened(event_names, boolean)
        for event_index, event in enumerate(trace):
            trace_row = []
            update_event_happened(event, event_names, event_happened, boolean)
            trace_row += event_happened

            trace_row.append(trace_name)
            # Start counting at 1
            trace_row.append(event_index + 1)
            trace_row.append(remaining_time(trace, event))
            trace_row.append(elapsed_time(trace, event))
            encoded_data.append(trace_row)

    return pd.DataFrame(columns=columns, data=encoded_data)


def create_event_happened(event_names: list, boolean: bool):
    """Creates list of event happened placeholders"""
    if boolean:
        return [False] * len(event_names)
    return [0] * len(event_names)


def update_event_happened(event, event_names: list, event_happened: list, boolean: bool):
    """Updates the event_happened list at event index

    For boolean set happened to True.
    For frequency updates happened count.
    """
    event_name = CLASSIFIER.get_class_identity(event)
    event_index = event_names.index(event_name)
    if boolean:
        event_happened[event_index] = True
    else:
        event_happened[event_index] += 1
