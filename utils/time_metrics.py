from datetime import datetime as dt

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


def duration(trace):
    """Calculate the duration of a trace"""
    return remaining_time_id(trace, 0)


def elapsed_time_id(trace, event_index: int):
    """Calculate elapsed time by event index in trace"""
    try:
        event = trace[event_index]
    except IndexError:
        # catch for 0 padding.
        # calculate using the last event in trace
        event = trace[-1]
    return elapsed_time(trace, event)


def elapsed_time(trace, event):
    """Calculate elapsed time by event in trace"""
    # FIXME using no timezone info for calculation
    event_time = event['time:timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    first_time = trace[0]['time:timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    try:
        delta = dt.strptime(event_time, TIME_FORMAT) - dt.strptime(first_time, TIME_FORMAT)
    except ValueError:
        # Log has no timestamps
        return 0
    return delta.total_seconds()


def remaining_time_id(trace, event_index: int):
    """Calculate remaining time by event index in trace"""
    try:
        event = trace[event_index]
        return remaining_time(trace, event)
    except IndexError:
        # catch for 0 padding.
        # cant calculate remaining time if there are no more events
        return 0


def remaining_time(trace, event):
    """Calculate remaining time by event in trace"""
    # FIXME using no timezone info for calculation
    event_time = event['time:timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    last_time = trace[-1]['time:timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    try:
        delta = dt.strptime(last_time, TIME_FORMAT) - dt.strptime(event_time, TIME_FORMAT)
    except ValueError:
        # Log has no timestamps
        return 0
    return delta.total_seconds()


def count_on_event_day(trace, date_dict: dict, event_id):
    """Finds the date of event and returns the value from date_dict
    :param date_dict one of the dicts from log_metrics.py
    :param event_id Event id
    :param trace Log trace
    """
    try:
        event = trace[event_id]
        date = event['time:timestamp'].date()
        return date_dict.get(date, 0)
    except IndexError:
        return 0
