import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from encoders.log_util import get_event_attributes
from .log_util import remaining_time_id, elapsed_time_id

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])


def complex_encode(data, event_names, prefix_length=1):
    if prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    return encode_complex(data, event_names, prefix_length, 'complex')


def last_payload(data, event_names, prefix_length=1):
    if prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    return encode_last_payload(data, event_names, prefix_length, 'latest')


def encode_complex(log: list, event_names: list, prefix_length: int, encoding='complex'):
    additional_columns = get_event_attributes(log)
    columns = columns_complex(prefix_length, additional_columns)
    encoded_data = []

    for trace in log:
        if len(trace) <= prefix_length - 1:
            continue
        trace_row = []
        trace_name = CLASSIFIER.get_class_identity(trace)
        trace_row.append(trace_name)
        # prefix_length - 1 == index
        trace_row.append(remaining_time_id(trace, prefix_length - 1))
        trace_row.append(elapsed_time_id(trace, prefix_length - 1))
        trace_row += trace_data_complex(trace, event_names, prefix_length, additional_columns)
        encoded_data.append(trace_row)

    return pd.DataFrame(columns=columns, data=encoded_data)


def encode_last_payload(log: list, event_names: list, prefix_length: int, encoding='complex'):
    additional_columns = get_event_attributes(log)
    columns = columns_last_payload(prefix_length, additional_columns)
    encoded_data = []

    for trace in log:
        if len(trace) <= prefix_length - 1:
            continue
        trace_row = []
        trace_name = CLASSIFIER.get_class_identity(trace)
        trace_row.append(trace_name)
        # prefix_length - 1 == index
        trace_row.append(remaining_time_id(trace, prefix_length - 1))
        trace_row.append(elapsed_time_id(trace, prefix_length - 1))
        trace_row += trace_data_last_payload(trace, event_names, prefix_length, additional_columns)
        encoded_data.append(trace_row)

    return pd.DataFrame(columns=columns, data=encoded_data)


def columns_complex(prefix_length, additional_columns):
    columns = ['trace_id', 'remaining_time', 'elapsed_time']
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
        for additional_column in additional_columns:
            columns.append(additional_column + "_" + str(i))
    return columns


def columns_last_payload(prefix_length, additional_columns):
    columns = ['trace_id', 'remaining_time', 'elapsed_time']
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
    for additional_column in additional_columns:
        columns.append(additional_column + "_" + str(i))
    return columns


def __add_case_data(df, case_events, case_data, additional_columns, encoding):
    if encoding == 'complex':
        return __complex_case_data(df, case_events, case_data, additional_columns)
    elif encoding == 'frequency':
        return __index_latest_case_data(df, case_events, case_data, additional_columns)


def __complex_case_data(df, case_events, case_data, additional_columns):
    for index in range(0, len(case_events)):
        case_data.append(case_events[index])
        __add_additional_columns(df, case_events, case_data, additional_columns)


def __index_latest_case_data(df, case_events, case_data, additional_columns):
    for index in range(0, len(case_events)):
        case_data.append(case_events[index])
    __add_additional_columns(df, case_events, case_data, additional_columns)


def __add_additional_columns(df, case_events, case_data, additional_columns):
    for additional_column in additional_columns:
        event_attribute = df[df['event_nr'] == len(case_events)][additional_column].apply(str).item()
        case_data.append(event_attribute)


def trace_data_complex(trace: list, event_names: list, prefix_length: int, additional_columns: list):
    """Creates list in form [1, value1, value2, 2, ...]

    Event name index of the position they are in event_names
    Appends values in additional_columns
    """
    data = list()
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = CLASSIFIER.get_class_identity(event)
        event_id = event_names.index(event_name)
        data.append(event_id + 1)  # prefix

        for att in additional_columns:
            # Basically XEventAttributeClassifier
            value = event.get_attributes().get(att).get_value()
            data.append(value)

    return data


def trace_data_last_payload(trace: list, event_names: list, prefix_length: int, additional_columns: list):
    """Creates list in form [1, 2, value1, value2,]

    Event name index of the position they are in event_names
    Appends values in additional_columns
    """
    data = list()
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = CLASSIFIER.get_class_identity(event)
        event_id = event_names.index(event_name)
        data.append(event_id + 1)  # prefix

    # Attributes of last event
    for att in additional_columns:
        # Basically XEventAttributeClassifier
        value = trace[prefix_length - 1].get_attributes().get(att).get_value()
        data.append(value)

    return data
