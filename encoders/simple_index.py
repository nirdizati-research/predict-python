import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from .log_util import remaining_time_id, elapsed_time_id, DEFAULT_COLUMNS

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])


def simple_index(log: list, event_names: list, prefix_length=1, next_activity=False):
    if next_activity:
        return encode_next_activity(log, event_names, prefix_length)
    return encode_simple_index(log, event_names, prefix_length)


def encode_simple_index(log: list, event_names: list, prefix_length: int, ):
    # Events up to prefix_length
    events_to_consider = event_names[:prefix_length]
    columns = __create_columns(prefix_length)
    encoded_data = []

    for trace in log:
        if len(trace) <= prefix_length:
            continue
        trace_row = []
        trace_name = CLASSIFIER.get_class_identity(trace)
        trace_row.append(trace_name)
        trace_row.append(prefix_length)
        trace_row.append(remaining_time_id(trace, prefix_length))
        trace_row.append(elapsed_time_id(trace, prefix_length))
        for idx, _ in enumerate(events_to_consider):
            trace_row.append(idx + 1)
        encoded_data.append(trace_row)

    return pd.DataFrame(columns=columns, data=encoded_data)


def encode_next_activity(log: list, event_names: list, prefix_length: int):
    columns = __columns_next_activity(prefix_length)
    encoded_data = []

    for trace in log:
        trace_row = []
        trace_name = CLASSIFIER.get_class_identity(trace)
        trace_row.append(trace_name)
        # trace_row.append(prefix_length)

        event_id = -100
        for idx, event in enumerate(trace):
            if idx == prefix_length:
                break
            event_name = CLASSIFIER.get_class_identity(event)
            event_id = event_names.index(event_name)
            trace_row.append(event_id + 1)

        print(trace_row)
        for _ in range(event_id + 1, prefix_length):
            trace_row.append(0)

        # event that is next
        next_event = trace[event_id + 1]
        next_event_name = CLASSIFIER.get_class_identity(next_event)
        print(next_event_name)
        next_event_id = event_names.index(next_event_name)
        print(next_event_id)
        trace_row.append(next_event_id + 1)
        encoded_data.append(trace_row)

    print(encoded_data)
    print(columns)
    return pd.DataFrame(columns=columns, data=encoded_data)


def __create_columns(prefix_length: int):
    columns = list(DEFAULT_COLUMNS)
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
    return columns


def __columns_next_activity(prefix_length):
    """Creates columns for next activity"""
    columns = ["trace_id"]
    for i in range(0, prefix_length):
        columns.append("prefix_" + str(i + 1))
    columns.append("label")
    return columns
