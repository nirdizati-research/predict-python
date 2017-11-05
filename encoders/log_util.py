from datetime import datetime as dt

from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
# TODO change to trace_id
DEFAULT_COLUMNS = ["case_id", "event_nr", "remaining_time", "elapsed_time"]
timestamp_classifier = XEventAttributeClassifier("Timestamp", ["time:timestamp"])


def unique_events(log: list):
    """List of unique events using event concept:name"""
    classifier = XEventAttributeClassifier("Resource", ["concept:name"])
    event_set = set()
    for trace in log:
        for event in trace:
            event_name = classifier.get_class_identity(event)
            event_set.add(event_name)
    return list(event_set)


def calculate_elapsed_time(trace, event_id):
    # FIXME using no timezone info for calculation
    event_time = timestamp_classifier.get_class_identity(trace[event_id])[:19]
    first_time = timestamp_classifier.get_class_identity(trace[0])[:19]
    delta = dt.strptime(event_time, TIME_FORMAT) - dt.strptime(first_time, TIME_FORMAT)
    return delta.total_seconds()


def calculate_remaining_time(trace, event_id):
    # FIXME using no timezone info for calculation
    event_time = timestamp_classifier.get_class_identity(trace[event_id])[:19]
    last_time = timestamp_classifier.get_class_identity(trace[-1])[:19]
    delta = dt.strptime(last_time, TIME_FORMAT) - dt.strptime(event_time, TIME_FORMAT)
    return delta.total_seconds()
