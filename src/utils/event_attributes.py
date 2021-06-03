from functools import reduce

from pm4py.objects.log.log import EventLog


def unique_events(log: EventLog):
    """List of unique events using event concept:name

    Adds all events into a list and removes duplicates while keeping order.
    :param log:
    :return:
    """

    event_list = [event['concept:name'] for trace in log for event in trace]
    # TODO: this is very strange
    return sorted(set(event_list), key=lambda x: event_list.index(x))


def unique_events2(training_log: EventLog, test_log: EventLog):
    """ Combines unique events from two logs into one list.

    Renamed to 2 because Python doesn't allow functions with same names.
    Python is objectively the worst language.
    :param training_log:
    :param test_log:
    :return:
    """
    event_list = unique_events(training_log) + unique_events(test_log)
    return sorted(set(event_list), key=lambda x: event_list.index(x))


def get_event_attributes(log: EventLog):
    """Get log event attributes that are not name or time

    As log file is a list, it has no global event attributes. Getting from first event of first trace. This may be bad.
    :param log:
    :return:
    """
    event_attributes = []
    for attribute in log[0][0]._dict.keys():
        if attribute not in ["concept:name", "time:timestamp"]:
            event_attributes.append(attribute)
    return sorted(event_attributes)


def get_additional_columns(log: EventLog):
    """Returns additional columns

    :param log:
    :return:
    """
    return {'trace_attributes': get_global_trace_attributes(log),
            'event_attributes': get_global_event_attributes(log)}


def get_global_trace_attributes(log: EventLog):
    """Get log trace attributes that are not name or time

    :param log:
    :return:
    """
    # retrieves all traces in the log and returns their intersection
    attributes = list(reduce(set.intersection, [set(trace._get_attributes().keys()) for trace in log]))
    trace_attributes = [attr for attr in attributes if attr not in ["concept:name", "time:timestamp", "label"]]
    return sorted(trace_attributes)


def get_global_event_attributes(log):
    """Get log event attributes that are not name or time

    :param log:
    :return:
    """
    # retrieves all events in the log and returns their intersection
    attributes = list(reduce(set.intersection, [set(event._dict.keys()) for trace in log for event in trace]))
    event_attributes = [attr for attr in attributes if attr not in ["concept:name", "time:timestamp"]]
    return sorted(event_attributes)
