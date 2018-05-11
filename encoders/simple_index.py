import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier
from opyenxes.model import XTrace

from encoders.label_container import *
from log_util.log_metrics import events_by_date, resources_by_date, new_trace_start
from log_util.time_metrics import duration, elapsed_time_id, remaining_time_id, count_on_event_day

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])
ATTRIBUTE_CLASSIFIER = None


def simple_index(log: list, label: LabelContainer, prefix_length=1, zero_padding=False, all_in_one=False):
    columns = __columns(prefix_length, label)
    encoded_data = []
    # Create classifier only once
    if label.type == ATTRIBUTE_STRING or label.type == ATTRIBUTE_NUMBER:
        global ATTRIBUTE_CLASSIFIER
        ATTRIBUTE_CLASSIFIER = XEventAttributeClassifier("Attr class", [label.attribute_name])
    # Expensive operations
    executed_events = events_by_date([log]) if label.add_executed_events else None
    resources_used = resources_by_date([log]) if label.add_resources_used else None
    new_traces = new_trace_start([log]) if label.add_new_traces else None
    add_features = {'executed_events': executed_events, 'resources_used': resources_used, 'new_traces': new_traces}
    for trace in log:
        if len(trace) <= prefix_length - 1 and not zero_padding:
            continue
        if all_in_one:
            for i in range(1, prefix_length + 1):
                encoded_data.append(
                    add_trace_row(trace, label, zero_padding, prefix_length, all_in_one, i, **add_features))
        else:
            encoded_data.append(add_trace_row(trace, label, zero_padding, prefix_length, all_in_one, prefix_length,
                                              **add_features))

    return pd.DataFrame(columns=columns, data=encoded_data)


def add_trace_row(trace: XTrace, label: LabelContainer, zero_padding: bool, prefix_length: int,
                  all_in_one: bool, event_index: int, executed_events=None, resources_used=None, new_traces=None):
    """Row in data frame"""
    if zero_padding:
        zero_count = event_index - len(trace)
    elif all_in_one:
        zero_count = prefix_length - event_index
    trace_row = list()
    trace_row.append(CLASSIFIER.get_class_identity(trace))
    trace_row += trace_prefixes(trace, event_index)
    if zero_padding or all_in_one:
        trace_row += ['0' for _ in range(0, zero_count)]
    trace_row += add_labels(label, event_index, trace, ATTRIBUTE_CLASSIFIER=ATTRIBUTE_CLASSIFIER,
                            executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
    return trace_row


def trace_prefixes(trace: list, prefix_length: int):
    """List of indexes of the position they are in event_names"""
    prefixes = list()
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = CLASSIFIER.get_class_identity(event)
        prefixes.append(event_name)
    return prefixes


def next_event_name(trace: list, prefix_length: int):
    """Return the event event name at prefix length
    Or '0' if out of range.
    """
    if prefix_length < len(trace):
        next_event = trace[prefix_length]
        name = CLASSIFIER.get_class_identity(next_event)
        return name
    else:
        return '0'


def __columns(prefix_length: int, label: LabelContainer):
    """trace_id, prefixes, any other columns, label"""
    columns = ["trace_id"]
    for i in range(0, prefix_length):
        columns.append("prefix_" + str(i + 1))
    return add_label_columns(columns, label)


def add_label_columns(columns: list, label: LabelContainer):
    if label.type == NO_LABEL:
        return columns
    if label.add_elapsed_time:
        columns.append('elapsed_time')
    if label.add_remaining_time and label.type != REMAINING_TIME:
        columns.append('remaining_time')
    if label.add_executed_events:
        columns.append('executed_events')
    if label.add_resources_used:
        columns.append('resources_used')
    if label.add_new_traces:
        columns.append('new_traces')
    columns.append('label')
    return columns


def add_labels(label: LabelContainer, prefix_length: int, trace,
               ATTRIBUTE_CLASSIFIER=ATTRIBUTE_CLASSIFIER, executed_events=None, resources_used=None, new_traces=None):
    """Adds any number of label cells with last as label"""
    labels = []
    if label.type == NO_LABEL:
        return labels
    # Values that can just be there
    if label.add_elapsed_time:
        labels.append(elapsed_time_id(trace, prefix_length - 1))
    if label.add_remaining_time and label.type != REMAINING_TIME:
        labels.append(remaining_time_id(trace, prefix_length - 1))
    if label.add_executed_events:
        labels.append(count_on_event_day(trace, executed_events, prefix_length - 1))
    if label.add_resources_used:
        labels.append(count_on_event_day(trace, resources_used, prefix_length - 1))
    if label.add_new_traces:
        labels.append(count_on_event_day(trace, new_traces, prefix_length - 1))
    # Label
    if label.type == REMAINING_TIME:
        labels.append(remaining_time_id(trace, prefix_length - 1))
    elif label.type == NEXT_ACTIVITY:
        labels.append(next_event_name(trace, prefix_length))
    elif label.type == ATTRIBUTE_STRING or label.type == ATTRIBUTE_NUMBER:
        atr = ATTRIBUTE_CLASSIFIER.get_class_identity(trace)
        labels.append(atr)
    elif label.type == DURATION:
        labels.append(duration(trace))
    return labels
