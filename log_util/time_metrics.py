from datetime import datetime as dt

from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
TIMESTAMP_CLASSIFIER = XEventAttributeClassifier("Timestamp", ["time:timestamp"])


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
    event_time = TIMESTAMP_CLASSIFIER.get_class_identity(event)[:19]
    first_time = TIMESTAMP_CLASSIFIER.get_class_identity(trace[0])[:19]
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
    event_time = TIMESTAMP_CLASSIFIER.get_class_identity(event)[:19]
    last_time = TIMESTAMP_CLASSIFIER.get_class_identity(trace[-1])[:19]
    try:
        delta = dt.strptime(last_time, TIME_FORMAT) - dt.strptime(event_time, TIME_FORMAT)
    except ValueError:
        # Log has no timestamps
        return 0
    return delta.total_seconds()
