from collections import defaultdict, OrderedDict

TIMESTAMP_CLASSIFIER = "time:timestamp"
NAME_CLASSIFIER = "concept:name"


def events_by_date(log):
    """Creates dict of events by date ordered by date

    :return {'2010-12-30': 7, '2011-01-06': 8}
    :rtype: OrderedDict
    """

    stamp_dict = defaultdict(lambda: 0)
    for trace in log:
        for event in trace:
            timestamp = event[TIMESTAMP_CLASSIFIER]
            stamp_dict[str(timestamp.date())] += 1
    return OrderedDict(sorted(stamp_dict.items()))


def resources_by_date(log):
    """Creates dict of used unique resources ordered by date

    Resource and timestamp delimited by &&. If this is in resources name, bad stuff will happen.
    Returns a dict with a date and the number of unique resources used on that day.
    :return {'2010-12-30': 7, '2011-01-06': 8}
    :rtype: OrderedDict
    """
    stamp_dict = defaultdict(lambda: [])
    for trace in log:
        for event in trace:
            resource = event.get("Resource", "")
            timestamp = event[TIMESTAMP_CLASSIFIER]
            stamp_dict[str(timestamp.date())].append(resource)

    for key, value in stamp_dict.items():
        stamp_dict[key] = len(set(value))

    return OrderedDict(sorted(stamp_dict.items()))


def event_executions(log):
    """Creates dict of event execution count

    :return {'Event A': 7, '2011-01-06': 8}
    :rtype: OrderedDict
    """
    executions = defaultdict(lambda: 0)
    for trace in log:
        for event in trace:
            executions[event[NAME_CLASSIFIER]] += 1
    return OrderedDict(sorted(executions.items()))


def new_trace_start(log):
    """Creates dict of new traces by date

    :return {'2010-12-30': 1, '2011-01-06': 2}
    :rtype: OrderedDict
    """
    executions = defaultdict(lambda: 0)
    for trace in log:
        timestamp = trace[0][TIMESTAMP_CLASSIFIER]
        executions[str(timestamp.date())] += 1
    return OrderedDict(sorted(executions.items()))


def trace_attributes(log):
    """Creates an array of dicts that describe trace attributes.
    Only looks at first trace. Filters out `concept:name`.

    :return [{name: 'name', type: 'string', example: 34}]
    :rtype list
    """
    values = []
    trace = log[0] #TODO: this might be a bug if first trace has different events then others
    for attribute in trace.attributes:
        if attribute != "concept:name":
            atr_type = is_number(trace.attributes[attribute])
            atr = {'name': attribute, 'type': atr_type, 'example': str(trace.attributes[attribute])}
            values.append(atr)
    values = sorted(values, key=lambda k: k['name'])
    return values


def is_number(s):
    if isinstance(s, float) or isinstance(s, int):
        return 'number'
    else:
        return 'string'


def events_in_trace(log):
    """Creates dict of number of events in trace

    :return {'4': 11, '3': 8}
    :rtype: OrderedDict
    """
    stamp_dict = defaultdict(lambda: 0)
    for trace in log:
        stamp_dict[trace[NAME_CLASSIFIER]] = len(trace)
    return OrderedDict(sorted(stamp_dict.items()))


def max_events_in_log(log):
    """Returns the maximum number of events in any trace

    :return 3
    :rtype: int
    """
    return max([len(trace) for trace in log])
