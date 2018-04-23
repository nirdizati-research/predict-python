import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from encoders.label_container import *
from .log_util import remaining_time_id, elapsed_time_id

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])
ATTRIBUTE_CLASSIFIER = None


def simple_index(log: list, event_names: list, label: LabelContainer, prefix_length=1, zero_padding=False):
    if prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    return encode_simple_index(log, event_names, prefix_length, label, zero_padding)


def encode_simple_index(log: list, event_names: list, prefix_length: int, label: LabelContainer, zero_padding: bool):
    columns = __columns(prefix_length, label)
    encoded_data = []

    # Create classifier only once
    if label.type == ATTRIBUTE_STRING or label.type == ATTRIBUTE_NUMBER:
        global ATTRIBUTE_CLASSIFIER
        ATTRIBUTE_CLASSIFIER = XEventAttributeClassifier("Attr class", [label.attribute_name])
    for trace in log:
        if zero_padding:
            zero_count = prefix_length - len(trace)
        elif len(trace) <= prefix_length - 1:
            # no padding, skip this trace
            continue
        trace_row = []
        trace_name = CLASSIFIER.get_class_identity(trace)
        trace_row.append(trace_name)
        trace_row += trace_prefixes(trace, event_names, prefix_length)
        if zero_padding:
            trace_row += [0 for _ in range(0, zero_count)]
        trace_row += add_labels(label, prefix_length, trace, event_names, ATTRIBUTE_CLASSIFIER=ATTRIBUTE_CLASSIFIER)
        encoded_data.append(trace_row)
    return pd.DataFrame(columns=columns, data=encoded_data)


def trace_prefixes(trace: list, event_names: list, prefix_length: int):
    """List of indexes of the position they are in event_names"""
    prefixes = list()
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = CLASSIFIER.get_class_identity(event)
        event_id = event_names.index(event_name)
        prefixes.append(event_id + 1)
    return prefixes


def next_event_index(trace: list, event_names: list, prefix_length: int):
    """Return the event_name index of the one after at prefix_length.
    Offset by +1.
    Or 0 if out of range.
    """
    if prefix_length < len(trace):
        next_event = trace[prefix_length]
        next_event_name = CLASSIFIER.get_class_identity(next_event)
        return event_names.index(next_event_name) + 1
    else:
        return 0


def __columns(prefix_length: int, label: LabelContainer):
    """trace_id, prefixes, any other columns, label"""
    columns = ["trace_id"]
    for i in range(0, prefix_length):
        columns.append("prefix_" + str(i + 1))
    return add_label_columns(columns, label)


def add_label_columns(columns: list, label: LabelContainer):
    if label.type == NO_LABEL:
        return columns
    if label.add_elapsed_time:
        columns.append('elapsed_time')
    if label.add_remaining_time and label.type != REMAINING_TIME:
        columns.append('remaining_time')
    columns.append('label')
    return columns


def add_labels(label: LabelContainer, prefix_length: int, trace, event_names: list,
               ATTRIBUTE_CLASSIFIER=ATTRIBUTE_CLASSIFIER):
    """Adds any number of label cells with last as label"""
    labels = []
    if label.type == NO_LABEL:
        return labels
    # Values that can just be there
    if label.add_elapsed_time:
        labels.append(elapsed_time_id(trace, prefix_length - 1))
    if label.add_remaining_time and label.type != REMAINING_TIME:
        labels.append(remaining_time_id(trace, prefix_length - 1))
    # Label
    if label.type == REMAINING_TIME:
        labels.append(remaining_time_id(trace, prefix_length - 1))
    elif label.type == NEXT_ACTIVITY:
        labels.append(next_event_index(trace, event_names, prefix_length))
    elif label.type == ATTRIBUTE_STRING or label.type == ATTRIBUTE_NUMBER:
        atr = ATTRIBUTE_CLASSIFIER.get_class_identity(trace)
        labels.append(atr)
    return labels
