from collections import defaultdict, OrderedDict

TIMESTAMP_CLASSIFIER = "time:timestamp"
NAME_CLASSIFIER = "concept:name"


def events_by_date(logs):
    """Creates dict of events by date ordered by date

    :return {'2010-12-30': 7, '2011-01-06': 8}
    :rtype: OrderedDict
    """

    stamp_dict = defaultdict(lambda: 0)
    for log in logs:
        for trace in log:
            for event in trace:
                timestamp = event[TIMESTAMP_CLASSIFIER]
                date = timestamp.split("T")[0]
                stamp_dict[date] += 1
    return OrderedDict(sorted(stamp_dict.items()))


def resources_by_date(logs):
    """Creates dict of used unique resources ordered by date

    Resource and timestamp delimited by &&. If this is in resources name, bad stuff will happen.
    Returns a dict with a date and the number of unique resources used on that day.
    :return {'2010-12-30': 7, '2011-01-06': 8}
    :rtype: OrderedDict
    """
    stamp_dict = defaultdict(lambda: [])
    for log in logs:
        for trace in log:
            for event in trace:
                resource = event["Resource"]
                timestamp = event["time:timestamp"]
                date = timestamp.split("T")[0]
                stamp_dict[date].append(resource)

    for key, value in stamp_dict.items():
        stamp_dict[key] = len(set(value))

    return OrderedDict(sorted(stamp_dict.items()))


def event_executions(logs):
    """Creates dict of event execution count

    :return {'Event A': 7, '2011-01-06': 8}
    :rtype: OrderedDict
    """
    executions = defaultdict(lambda: 0)
    for log in logs:
        for trace in log:
            for event in trace:
                event_name = event[NAME_CLASSIFIER]
                executions[event_name] += 1
    return OrderedDict(sorted(executions.items()))


def new_trace_start(logs):
    """Creates dict of new traces by date

    :return {'2010-12-30': 1, '2011-01-06': 2}
    :rtype: OrderedDict
    """
    executions = defaultdict(lambda: 0)
    for log in logs:
        for trace in log:
            timestamp = trace[0][TIMESTAMP_CLASSIFIER]
            date = timestamp.split("T")[0]
            executions[date] += 1
    return OrderedDict(sorted(executions.items()))


def trace_attributes(logs):
    """Creates an array of dicts that describe trace attributes.
    Only looks at first trace. Filters out `concept:name`.

    :return [{name: 'name', type: 'string', example: 34}]
    :rtype list
    """
    values = []
    for log in logs:
        trace = log[0]
        for attribute in trace._get_attributes().values():
            if attribute.get_key() != "concept:name":
                atr_type = is_number(attribute.get_value())
                atr = {'name': attribute.get_key(), 'type': atr_type, 'example': str(attribute.get_value())}
                values.append(atr)
    values = sorted(values, key=lambda k: k['name'])
    return values


def is_number(s):
    try:
        float(s)
        return 'number'
    except Exception:
        return 'string'


def events_in_trace(logs):
    """Creates dict of number of events in trace

    :return {'4': 11, '3': 8}
    :rtype: OrderedDict
    """
    stamp_dict = defaultdict(lambda: 0)
    for log in logs:
        for trace in log:
            counter = 0
            for event in trace:
                counter += 1
            name = trace[NAME_CLASSIFIER]
            stamp_dict[name] = counter
    return OrderedDict(sorted(stamp_dict.items()))


def max_events_in_log(logs):
    """Returns the maximum number of events in any trace

    :return 3
    :rtype: int
    """
    current_max = 0
    for log in logs:
        for trace in log:
            counter = len(trace)

            if counter > current_max:
                current_max = counter

    return current_max
