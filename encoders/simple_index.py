import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from encoders.log_util import HEADER_COLUMNS
from .log_util import remaining_time_id, elapsed_time_id

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])


def simple_index(log: list, event_names: list, prefix_length=1, next_activity=False, add_label=True,
                 zero_padding=False):
    if prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    if next_activity:
        return encode_next_activity(log, event_names, prefix_length, add_label, zero_padding)
    return encode_simple_index(log, event_names, prefix_length, add_label, zero_padding)


def encode_simple_index(log: list, event_names: list, prefix_length: int, add_label: bool, zero_padding: bool):
    columns = __create_columns(prefix_length, add_label)
    encoded_data = []

    for trace in log:
        if zero_padding:
            zero_count = prefix_length - len(trace)
        elif len(trace) <= prefix_length - 1:
            # no padding, skip this trace
            continue
        trace_row = []
        trace_name = CLASSIFIER.get_class_identity(trace)
        trace_row.append(trace_name)
        if add_label:
            trace_row.append(remaining_time_id(trace, prefix_length - 1))
        trace_row.append(elapsed_time_id(trace, prefix_length - 1))
        trace_row += trace_prefixes(trace, event_names, prefix_length)
        if zero_padding:
            trace_row += [0 for _ in range(0, zero_count)]
        encoded_data.append(trace_row)
    return pd.DataFrame(columns=columns, data=encoded_data)


def encode_next_activity(log: list, event_names: list, prefix_length: int, add_label: bool, zero_padding: bool):
    columns = __columns_next_activity(prefix_length, add_label)
    encoded_data = []

    for trace in log:
        if not zero_padding and len(trace) <= prefix_length - 1:
            # no padding, skip this trace
            continue
        trace_row = []
        trace_name = CLASSIFIER.get_class_identity(trace)
        trace_row.append(trace_name)

        trace_row += trace_prefixes(trace, event_names, prefix_length)

        for _ in range(len(trace), prefix_length):
            trace_row.append(0)

        if add_label:
            trace_row.append(next_event_index(trace, event_names, prefix_length))
        encoded_data.append(trace_row)

    return pd.DataFrame(columns=columns, data=encoded_data)


def __create_columns(prefix_length: int, add_label: bool):
    if add_label:
        columns = list(HEADER_COLUMNS)
    else:
        columns = ['trace_id', 'elapsed_time']

    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
    return columns


def __columns_next_activity(prefix_length: int, add_label: bool):
    """Creates columns for next activity"""
    columns = ["trace_id"]
    for i in range(0, prefix_length):
        columns.append("prefix_" + str(i + 1))
    if add_label:
        columns.append("label")
    return columns


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
