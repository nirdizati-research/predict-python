import statistics
from collections import defaultdict, OrderedDict

from pm4py.objects.log.log import EventLog

TIMESTAMP_CLASSIFIER = "time:timestamp"
NAME_CLASSIFIER = "concept:name"


def events_by_date(log: EventLog) -> OrderedDict:
    """Creates dict of events by date ordered by date

    :param log:
    :return {'2010-12-30': 7, '2011-01-06': 8}
    :rtype: OrderedDict
    """

    stamp_dict = defaultdict(lambda: 0)
    for trace in log:
        for event in trace:
            timestamp = event[TIMESTAMP_CLASSIFIER]
            stamp_dict[str(timestamp.date())] += 1
    return OrderedDict(sorted(stamp_dict.items()))


def resources_by_date(log: EventLog) -> OrderedDict:
    """Creates dict of used unique resources ordered by date

    Resource and timestamp delimited by &&. If this is in resources name, bad stuff will happen.
    Returns a dict with a date and the number of unique resources used on that day.

    :param log:
    :return {'2010-12-30': 7, '2011-01-06': 8}
    """
    stamp_dict = defaultdict(lambda: [])
    for trace in log:
        for event in trace:
            resource = event.get("Resource", event.get("org:resource", ""))
            timestamp = event[TIMESTAMP_CLASSIFIER]
            stamp_dict[str(timestamp.date())].append(resource)

    for key, value in stamp_dict.items():
        stamp_dict[key] = len(set(value))

    return OrderedDict(sorted(stamp_dict.items()))


def event_executions(log: EventLog) -> OrderedDict:
    """Creates dict of event execution count

    :param log:
    :return {'Event A': 7, '2011-01-06': 8}
    """
    executions = defaultdict(lambda: 0)
    for trace in log:
        for event in trace:
            executions[event[NAME_CLASSIFIER]] += 1
    return OrderedDict(sorted(executions.items()))


def new_trace_start(log: EventLog) -> OrderedDict:
    """Creates dict of new traces by date

    :param log:
    :return {'2010-12-30': 1, '2011-01-06': 2}
    """
    executions = defaultdict(lambda: 0)
    for trace in log:
        timestamp = trace[0][TIMESTAMP_CLASSIFIER]
        executions[str(timestamp.date())] += 1
    return OrderedDict(sorted(executions.items()))


def trace_attributes(log: EventLog) -> list:
    """Creates an array of dicts that describe trace attributes.
    Only looks at first trace. Filters out `concept:name`.

    :param log:
    :return [{name: 'name', type: 'string', example: 34}]
    """
    values = []
    trace = log[0]  # TODO: this might be a bug if first trace has different events then others
    for attribute in trace.attributes:
        if attribute != "concept:name":
            atr_type = _is_number(trace.attributes[attribute])
            atr = {'name': attribute, 'type': atr_type, 'example': str(trace.attributes[attribute])}
            values.append(atr)
    values = sorted(values, key=lambda k: k['name'])
    return values


def _is_number(s) -> str:
    """Returns whether the parameter is a number or string

    :param s:
    :return:
    """
    if (isinstance(s, (float, int)) or (s.isdigit() if hasattr(s, 'isdigit') else False)) and not isinstance(s, bool):
        return 'number'
    return 'string'


def events_in_trace(log: EventLog) -> OrderedDict:
    """Creates dict of number of events in trace

    :param log:
    :return {'4': 11, '3': 8}
    """
    stamp_dict = defaultdict(lambda: 0)
    for trace in log:
        stamp_dict[trace.attributes[NAME_CLASSIFIER]] = len(trace)
    return OrderedDict(sorted(stamp_dict.items()))


def max_events_in_log(log: EventLog) -> int:
    """Returns the maximum number of events in any trace

    :param log:
    :return 3
    """
    return max([len(trace) for trace in log])


def avg_events_in_log(log: EventLog) -> int:
    """Returns the average number of events in any trace

    :param log:
    :return 3
    """
    return statistics.mean([len(trace) for trace in log])


def std_var_events_in_log(log: EventLog) -> int:
    """Returns the standard variation of the average number of events in any trace

    :param log:
    :return 3
    """
    return statistics.stdev([len(trace) for trace in log])


def trace_ids_in_log(log: EventLog) -> list:
    """Returns a list of trace's name classifier in the given log

    :param log:
    :return:
    """
    return [trace.attributes[NAME_CLASSIFIER] for trace in log]


def traces_in_log(log: EventLog) -> list:
    """Returns a list of dict, of traces in the given log

    :param log:
    :return:
    """
    return [{'attributes': trace.attributes, 'events': [event for event in trace]} for trace in log]


