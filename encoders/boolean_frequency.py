import numpy as np
import pandas as pd

from encoders.log_util import DEFAULT_COLUMNS


def boolean(data):
    return encode_boolean_frequency(data, 'boolean')


def frequency(data):
    return encode_boolean_frequency(data, 'frequency')


def encode_boolean_frequency(data, encoding='boolean'):
    """Internal method for both boolean and frequency. Only dif is __append_item"""
    events = get_events(data)
    case_ids = get_cases(data)

    columns = np.append(events, list(DEFAULT_COLUMNS))
    encoded_data = []

    for case_id in case_ids:
        case = data[data['case_id'] == case_id]
        for event_length in range(1, max(case['event_nr']) + 1):
            case_data = []
            for event in events:
                case_data.append(__append_item(case, event, event_length, encoding))
            case_data.append(case_id)
            case_data.append(event_length)
            remaining_time = calculate_remaining_time(case, event_length)
            case_data.append(remaining_time)
            elapsed_time = calculate_elapsed_time(case, event_length)
            case_data.append(elapsed_time)
            encoded_data.append(case_data)

    return pd.DataFrame(columns=columns, data=encoded_data)


def __append_item(df, event, event_length, encoding):
    """Boolean returns if len is > 0, frequency returns len"""
    length = len(df[(df['activity_name'] == event) & (df['event_nr'] <= event_length)])
    if encoding == 'boolean':
        return length > 0
    elif encoding == 'frequency':
        return length
