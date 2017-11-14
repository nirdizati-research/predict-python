from datetime import datetime as dt

from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DEFAULT_COLUMNS = ["trace_id", "event_nr", "remaining_time", "elapsed_time"]
TIMESTAMP_CLASSIFIER = XEventAttributeClassifier("Timestamp", ["time:timestamp"])


def unique_events(log: list):
    """List of unique events using event concept:name

    Adds all events into a list and removes duplicates while keeping order.
    """
    classifier = XEventAttributeClassifier("Resource", ["concept:name"])
    event_list = []
    for trace in log:
        for event in trace:
            event_name = classifier.get_class_identity(event)
            event_list.append(event_name)
    return sorted(set(event_list), key=lambda x: event_list.index(x))


def elapsed_time_id(trace, event_index: int):
    """Calculate elapsed time by event index in trace"""
    return elapsed_time(trace, trace[event_index])


def elapsed_time(trace, event):
    """Calculate elapsed time by event in trace"""
    # FIXME using no timezone info for calculation
    event_time = TIMESTAMP_CLASSIFIER.get_class_identity(event)[:19]
    first_time = TIMESTAMP_CLASSIFIER.get_class_identity(trace[0])[:19]
    delta = dt.strptime(event_time, TIME_FORMAT) - dt.strptime(first_time, TIME_FORMAT)
    return delta.total_seconds()


def remaining_time_id(trace, event_index: int):
    """Calculate remaining time by event index in trace"""
    return remaining_time(trace, trace[event_index])


def remaining_time(trace, event):
    """Calculate remaining time by event in trace"""
    # FIXME using no timezone info for calculation
    event_time = TIMESTAMP_CLASSIFIER.get_class_identity(event)[:19]
    last_time = TIMESTAMP_CLASSIFIER.get_class_identity(trace[-1])[:19]
    delta = dt.strptime(last_time, TIME_FORMAT) - dt.strptime(event_time, TIME_FORMAT)
    return delta.total_seconds()
