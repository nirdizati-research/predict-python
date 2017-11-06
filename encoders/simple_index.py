import pandas as pd

from .log_util import *

classifier = XEventAttributeClassifier("Trace name", ["concept:name"])


def encode_trace(data, prefix_length=1, next_activity=False):
    if next_activity:
        return encode_next_activity(data, prefix_length)
    return encode_simple_index(data, prefix_length)


def encode_simple_index(log: list, prefix_length: int):
    # Events up to prefix_length
    events_to_consider = unique_events(log)[:prefix_length]
    columns = __create_columns(prefix_length)
    encoded_data = []

    for trace in log:
        trace_row = []
        trace_name = classifier.get_class_identity(trace)
        trace_row.append(trace_name)
        trace_row.append(prefix_length)
        remaining_time = remaining_time_id(trace, prefix_length)
        trace_row.append(remaining_time)
        elapsed_time = elapsed_time_id(trace, prefix_length)
        trace_row.append(elapsed_time)
        for idx, event in enumerate(events_to_consider):
            trace_row.append(idx + 1)
        encoded_data.append(trace_row)

    return pd.DataFrame(columns=columns, data=encoded_data)


def encode_next_activity(log: list, prefix_length: int):
    # Events up to prefix_length
    events_to_consider = unique_events(log)[:prefix_length]
    columns = __columns_next_activity(prefix_length)
    encoded_data = []

    for trace in log:
        trace_row = []
        trace_name = classifier.get_class_identity(trace)
        trace_row.append(trace_name)
        trace_row.append(prefix_length)

        for idx, event in enumerate(events_to_consider[:-1]):
            trace_row.append(idx + 1)
        for k in range(len(events_to_consider), prefix_length):
            trace_row.append(0)

        # last id of event
        trace_row.append(len(events_to_consider))
        encoded_data.append(trace_row)

    return pd.DataFrame(columns=columns, data=encoded_data)


def __create_columns(prefix_length: int):
    columns = list(DEFAULT_COLUMNS)
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
    return columns


def __columns_next_activity(prefix_length):
    """Creates columns for next activity"""
    columns = ["case_id", "event_nr"]
    for i in range(1, prefix_length):
        columns.append("prefix_" + str(i))
    columns.append("label")
    return columns
