import numpy as np
import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from encoders.label_container import LabelContainer, ATTRIBUTE_STRING, ATTRIBUTE_NUMBER
from encoders.simple_index import add_label_columns, add_labels
from log_util.log_metrics import events_by_date, resources_by_date, new_trace_start

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])
ATTRIBUTE_CLASSIFIER = None


def boolean(log: list, event_names: list, label: LabelContainer, prefix_length=1, zero_padding=False):
    if prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    return encode_boolean_frequency(log, event_names, label, prefix_length, zero_padding, is_boolean=True)


def frequency(log: list, event_names: list, label: LabelContainer, prefix_length=1, zero_padding=False):
    if prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    return encode_boolean_frequency(log, event_names, label, prefix_length, zero_padding, is_boolean=False)


def encode_boolean_frequency(log: list, event_names: list, label: LabelContainer, prefix_length: int,
                             zero_padding: bool, is_boolean=True):
    """Encodes the log by boolean or frequency

    trace_id, event_nr, event_names, label stuff
    :return pandas dataframe
    """
    columns = create_columns(event_names, label)
    encoded_data = []

    # Create classifier only once
    if label.type == ATTRIBUTE_STRING or label.type == ATTRIBUTE_NUMBER:
        global ATTRIBUTE_CLASSIFIER
        ATTRIBUTE_CLASSIFIER = XEventAttributeClassifier("Attr class", [label.attribute_name])
    # Expensive operations
    executed_events = events_by_date([log]) if label.add_executed_events else None
    resources_used = resources_by_date([log]) if label.add_resources_used else None
    new_traces = new_trace_start([log]) if label.add_new_traces else None
    for trace in log:
        if zero_padding:
            # zero padding happens by default
            pass
        elif len(trace) <= prefix_length - 1:
            # no padding, skip this trace
            continue
        # starts with all False, changes to event
        event_happened = create_event_happened(event_names, is_boolean)
        trace_row = []
        trace_name = CLASSIFIER.get_class_identity(trace)
        trace_row.append(trace_name)
        for event_index, event in enumerate(trace):
            if event_index >= prefix_length:
                pass
            else:
                update_event_happened(event, event_names, event_happened, is_boolean)
        trace_row += event_happened
        trace_row += add_labels(label, prefix_length, trace, event_names, ATTRIBUTE_CLASSIFIER=ATTRIBUTE_CLASSIFIER,
                                executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
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


def create_columns(event_names: list, label: LabelContainer):
    columns = ["trace_id"]
    columns = np.append(columns, event_names).tolist()
    return add_label_columns(columns, label)
