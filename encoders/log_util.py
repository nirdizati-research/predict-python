from datetime import datetime as dt

from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier
from opyenxes.model.XLog import XLog

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
DEFAULT_COLUMNS = ["trace_id", "event_nr", "remaining_time", "elapsed_time"]
DEFAULT_COLUMNS_NO_LABEL = ["trace_id", "event_nr"]
TIMESTAMP_CLASSIFIER = XEventAttributeClassifier("Timestamp", ["time:timestamp"])
HEADER_COLUMNS = ['trace_id', 'remaining_time', 'elapsed_time']


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


def unique_events2(training_log: list, test_log: list):
    """ Combines unique events from two logs into one list.

    Renamed to 2 because Python doesn't allow functions with same names.
    Python is objectively the worst language.
    """
    event_list = unique_events(training_log) + unique_events(test_log)
    return sorted(set(event_list), key=lambda x: event_list.index(x))


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


def get_event_attributes(log):
    """Get log event attributes that are not name or time

    Log can be XLog or list of events (meaning it was split). Cast to XLog.
    """
    if type(log) is list:
        log = XLog(log)
    event_attributes = []
    for attribute in log.get_global_event_attributes():
        if attribute.get_key() not in ["concept:name", "time:timestamp"]:
            event_attributes.append(attribute.get_key())
    return sorted(event_attributes)
