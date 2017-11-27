from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from encoders.log_util import unique_events, get_event_attributes
import pandas as pd
from .log_util import remaining_time_id, elapsed_time_id

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])


def complex_encode(data, event_names, prefix_length=1):
    return encode_complex_index_latest(data, event_names, prefix_length, 'complex')


def last_index(data, event_names, prefix_length=1):
    return encode_complex_index_latest(data, event_names, prefix_length, 'latest')


def encode_complex_index_latest(log: list, event_names: list, prefix_length: int, encoding='complex'):
    additional_columns = get_event_attributes(log)
    columns = __create_columns(prefix_length, additional_columns, encoding)
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
        trace_row += trace_data(trace, event_names, prefix_length, additional_columns)
        encoded_data.append(trace_row)

    return pd.DataFrame(columns=columns, data=encoded_data)


def encode_complex_index_latest2(data, additional_columns, prefix_length=1, encoding='complex'):
    """Internal method for complex and index latest encoding.
        Diff in columns and case_data.
    """
    case_ids = unique_events(data)
    columns = __create_columns(prefix_length, additional_columns, encoding)
    encoded_data = []

    for case_id in case_ids:
        case = data[data['case_id'] == case_id]
        event_length = prefix_length
        # uncomment to encode whole log at each prefix
        # for event_length in range(1, prefix_length+1):
        case_data = []
        case_data.append(case_id)
        case_data.append(event_length)
        remaining_time = remaining_time_id(case, event_length)
        case_data.append(remaining_time)
        elapsed_time = elapsed_time_id(case, event_length)
        case_data.append(elapsed_time)

        case_events = case[case['event_nr'] <= event_length]['activity_name'].tolist()
        __add_case_data(case, case_events, case_data, additional_columns, encoding)

        encoded_data.append(case_data)

    df = pd.DataFrame(columns=columns, data=encoded_data)
    return df


def __create_columns(prefix_length, additional_columns, encoding):
    if encoding == 'complex':
        return __create_complex_columns(prefix_length, additional_columns)
    elif encoding == 'frequency':
        return __create_index_latest_columns(prefix_length, additional_columns)


def __create_complex_columns(prefix_length, additional_columns):
    columns = ['trace_id', 'remaining_time', 'elapsed_time']
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
        for additional_column in additional_columns:
            columns.append(additional_column + "_" + str(i))
    return columns


def __create_index_latest_columns(prefix_length, additional_columns):
    max_length = prefix_length + 1
    columns = list(DEFAULT_COLUMNS2)
    for i in range(1, max_length):
        columns.append("prefix_" + str(i))
    for additional_column in additional_columns:
        columns.append(additional_column)
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


def trace_data(trace: list, event_names: list, prefix_length: int, additional_columns: list):
    """Creates list in form [1, value1, value2, 2, ...]

    Event name index of the position they are in event_names
    Appends values in additional_columns
    """
    prefixes = list()
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = CLASSIFIER.get_class_identity(event)
        event_id = event_names.index(event_name)
        prefixes.append(event_id + 1)

        for att in additional_columns:
            # Basically XEventAttributeClassifier
            value = event.get_attributes().get(att).get_value()
            prefixes.append(value)

    return prefixes
