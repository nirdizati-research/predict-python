from datetime import datetime as dt

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_COLUMNS = ["case_id", "event_nr", "remaining_time", "elapsed_time"]


def get_events(df):
    return df['activity_name'].unique()


def get_cases(df):
    return df['case_id'].unique()


def calculate_remaining_time(trace, event_nr):
    event_timestamp = trace[trace["event_nr"] == event_nr]['time'].apply(str).item()
    event_timestamp = dt.strptime(event_timestamp, TIME_FORMAT)
    last_event_timestamp = trace[trace["event_nr"] == len(trace)]['time'].apply(str).item()
    last_event_timestamp = dt.strptime(last_event_timestamp, TIME_FORMAT)
    return (last_event_timestamp - event_timestamp).total_seconds()


def calculate_elapsed_time(trace, event_nr):
    event_timestamp = trace[trace["event_nr"] == event_nr]['time'].apply(str).item()
    event_timestamp = dt.strptime(event_timestamp, TIME_FORMAT)
    first_event_timestamp = trace[trace["event_nr"] == 1]['time'].apply(str).item()
    first_event_timestamp = dt.strptime(first_event_timestamp, TIME_FORMAT)
    return (event_timestamp - first_event_timestamp).total_seconds()
