from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier
from opyenxes.model.XLog import XLog


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


def get_event_attributes(log: list):
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
