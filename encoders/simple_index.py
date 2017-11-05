import pandas as pd

from .helper import *


def encode_trace(data, prefix_length=1, next_activity=False):
    if next_activity:
        return __encode_next_activity(data, prefix_length)
    return __encode_simple(data, prefix_length)


def __encode_simple(data, prefix_length: int):
    events = get_events(data).tolist()
    case_ids = get_cases(data)

    columns = __create_columns(prefix_length)
    encoded_data = []

    for case_id in case_ids:
        case = data[data['case_id'] == case_id]
        if max(case['event_nr']) < prefix_length:
            # Skip greater values, otherwise index out of bounds
            continue
        event_length = prefix_length
        case_data = []
        case_data.append(case_id)
        case_data.append(event_length)
        remaining_time = calculate_remaining_time(case, event_length)
        case_data.append(remaining_time)
        elapsed_time = calculate_elapsed_time(case, event_length)
        case_data.append(elapsed_time)

        case_events = case[case['event_nr'] <= event_length]['activity_name'].tolist()
        for e in case_events:
            case_data.append(events.index(e) + 1)
        encoded_data.append(case_data)

    return pd.DataFrame(columns=columns, data=encoded_data)


def __encode_next_activity(data, prefix_length: int):
    events = get_events(data).tolist()
    case_ids = get_cases(data)

    columns = __columns_next_activity(prefix_length)
    encoded_data = []

    for case_id in case_ids:
        case = data[data['case_id'] == case_id]
        event_length = prefix_length
        case_data = list()
        case_data.append(case_id)
        case_data.append(event_length)

        case_events = case[case['event_nr'] <= event_length]['activity_name'].tolist()
        for e in case_events[:-1]:
            case_data.append(events.index(e) + 1)

        for k in range(len(case_events), prefix_length):
            case_data.append(0)

        label = events.index(case_events[-1]) + 1
        case_data.append(label)
        encoded_data.append(case_data)

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
