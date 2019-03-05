from pm4py.objects.log.log import EventLog

from src.encoding.models import Encoding
from src.labelling.models import LabelTypes, Labelling
from src.utils.log_metrics import events_by_date, resources_by_date, new_trace_start
from src.utils.time_metrics import elapsed_time_id, remaining_time_id, count_on_event_day, duration


def get_intercase_attributes(log: EventLog, encoding: Encoding):
    """Dict of kwargs
    These intercase attributes are expensive operations!!!
    """
    # Expensive operations
    executed_events = events_by_date(log) if encoding.add_executed_events else None
    resources_used = resources_by_date(log) if encoding.add_resources_used else None
    new_traces = new_trace_start(log) if encoding.add_new_traces else None
    kwargs = {'executed_events': executed_events, 'resources_used': resources_used, 'new_traces': new_traces}
    # 'label': label}  TODO: is it really necessary to add this field in the dict?
    return kwargs


def compute_label_columns(columns: list, encoding: Encoding, labelling: Labelling) -> list:
    if labelling.type == LabelTypes.NO_LABEL.value:
        return columns
    if encoding.add_elapsed_time:
        columns.append('elapsed_time')
    if encoding.add_remaining_time and labelling.type != LabelTypes.REMAINING_TIME.value:
        columns.append('remaining_time')
    if encoding.add_executed_events:
        columns.append('executed_events')
    if encoding.add_resources_used:
        columns.append('resources_used')
    if encoding.add_new_traces:
        columns.append('new_traces')
    columns.append('label')
    return columns


def add_labels(encoding: Encoding, labelling: Labelling, prefix_length: int, trace, attribute_classifier=None,
               executed_events=None, resources_used=None, new_traces=None):
    """
    Adds any number of label cells with last as label
    """
    labels = []
    if labelling.type == LabelTypes.NO_LABEL.value:
        return labels
    # Values that can just be there
    if encoding.add_elapsed_time:
        labels.append(elapsed_time_id(trace, prefix_length - 1))
    if encoding.add_remaining_time and labelling.type != LabelTypes.REMAINING_TIME.value:
        labels.append(remaining_time_id(trace, prefix_length - 1))
    if encoding.add_executed_events:
        labels.append(count_on_event_day(trace, executed_events, prefix_length - 1))
    if encoding.add_resources_used:
        labels.append(count_on_event_day(trace, resources_used, prefix_length - 1))
    if encoding.add_new_traces:
        labels.append(count_on_event_day(trace, new_traces, prefix_length - 1))
    # Label
    if labelling.type == LabelTypes.REMAINING_TIME.value:
        labels.append(remaining_time_id(trace, prefix_length - 1))
    elif labelling.type == LabelTypes.NEXT_ACTIVITY.value:
        labels.append(next_event_name(trace, prefix_length))
    elif labelling.type == LabelTypes.ATTRIBUTE_STRING.value or labelling.type == LabelTypes.ATTRIBUTE_NUMBER.value:
        labels.append(trace.attributes[attribute_classifier])
    elif labelling.type == LabelTypes.DURATION.value:
        labels.append(duration(trace))
    return labels


def next_event_name(trace: list, prefix_length: int):
    """Return the event event name at prefix length or 0 if out of range.

    """
    if prefix_length < len(trace):
        next_event = trace[prefix_length]
        name = next_event['concept:name']
        return name
    else:
        return 0
