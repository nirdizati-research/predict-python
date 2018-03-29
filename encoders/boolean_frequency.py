import numpy as np
import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from encoders.log_util import DEFAULT_COLUMNS, remaining_time, elapsed_time

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])


def boolean(log: list, event_names: list):
    return encode_boolean_frequency(log, event_names, is_boolean=True)


def frequency(log: list, event_names: list):
    return encode_boolean_frequency(log, event_names, is_boolean=False)


def encode_boolean_frequency(log: list, event_names: list, is_boolean=True):
    """Encodes the log by boolean or frequency

    :return pandas dataframe
    """
    columns = np.append(event_names, list(DEFAULT_COLUMNS))
    encoded_data = []

    for trace in log:
        trace_name = CLASSIFIER.get_class_identity(trace)
        # starts with all False, changes to event
        event_happened = create_event_happened(event_names, is_boolean)
        for event_index, event in enumerate(trace):
            trace_row = []
            update_event_happened(event, event_names, event_happened, is_boolean)
            trace_row += event_happened

            trace_row.append(trace_name)
            # Start counting at 1
            trace_row.append(event_index + 1)
            trace_row.append(remaining_time(trace, event))
            trace_row.append(elapsed_time(trace, event))
            encoded_data.append(trace_row)
    return pd.DataFrame(columns=columns, data=encoded_data)


def create_event_happened(event_names: list, is_boolean: bool):
    """Creates list of event happened placeholders"""
    if is_boolean:
        return [False] * len(event_names)
    return [0] * len(event_names)


def update_event_happened(event, event_names: list, event_happened: list, is_boolean: bool):
    """Updates the event_happened list at event index

    For boolean set happened to True.
    For frequency updates happened count.
    """
    event_name = CLASSIFIER.get_class_identity(event)
    event_index = event_names.index(event_name)
    if is_boolean:
        event_happened[event_index] = True
    else:
        event_happened[event_index] += 1
