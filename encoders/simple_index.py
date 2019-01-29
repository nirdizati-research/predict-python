import pandas as pd
from opyenxes.model import XTrace

from encoders.encoding_container import EncodingContainer
from encoders.label_container import *
from log_util.log_metrics import events_by_date, resources_by_date, new_trace_start
from log_util.time_metrics import duration, elapsed_time_id, remaining_time_id, count_on_event_day

ATTRIBUTE_CLASSIFIER = None


def simple_index(log: list, label: LabelContainer, encoding: EncodingContainer):
    columns = __columns(encoding.prefix_length, label)
    encoded_data = []
    kwargs = get_intercase_attributes(log, label)
    for trace in log:
        if len(trace) <= encoding.prefix_length - 1 and not encoding.is_zero_padding():
            # trace too short and no zero padding
            continue
        if encoding.is_all_in_one():
            for i in range(1, min(encoding.prefix_length + 1, len(trace) + 1)):
                encoded_data.append(add_trace_row(trace, encoding, i, label.attribute_name, **kwargs))
        else:
            encoded_data.append(add_trace_row(trace, encoding, encoding.prefix_length, label.attribute_name, **kwargs))

    return pd.DataFrame(columns=columns, data=encoded_data)


def add_trace_row(trace: XTrace, encoding: EncodingContainer, event_index: int, atr_classifier=None, label=None,
                  executed_events=None, resources_used=None, new_traces=None):
    """Row in data frame"""
    # a and b are magic values
    b = encoding.prefix_length - len(trace)
    if encoding.is_all_in_one():
        a = encoding.prefix_length - event_index
        zero_count = a if a > b else b
    elif encoding.is_zero_padding():
        zero_count = b
    else:
        print('encoding neither all_in_one nor zero_padding, setting zero count to 0!')
        zero_count = 0
    trace_row = list()
    trace_row.append(trace.attributes['concept:name'])
    trace_row += trace_prefixes(trace, event_index)
    if encoding.is_zero_padding() or encoding.is_all_in_one():
        trace_row += ['0' for _ in range(0, zero_count)]
    trace_row += add_labels(label, event_index, trace, atr_classifier=atr_classifier,
                            executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
    return trace_row


def trace_prefixes(trace: list, prefix_length: int):
    """List of indexes of the position they are in event_names"""
    prefixes = list()
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = event['concept:name']
        prefixes.append(event_name)
    return prefixes


def next_event_name(trace: list, prefix_length: int):
    """Return the event event name at prefix length
    Or '0' if out of range.
    """
    if prefix_length < len(trace):
        next_event = trace[prefix_length]
        name = next_event['concept:name']
        return name
    else:
        return '0'


def __columns(prefix_length: int, label: LabelContainer):
    """trace_id, prefixes, any other columns, label"""
    columns = ["trace_id"]
    for i in range(0, prefix_length):
        columns.append("prefix_" + str(i + 1))
    return add_label_columns(columns, label)


def get_intercase_attributes(log: list, label: LabelContainer):
    """Dict of kwargs
    These intercae attributes are expensive operations!!!
    """
    # Expensive operations
    executed_events = events_by_date([log]) if label.add_executed_events else None
    resources_used = resources_by_date([log]) if label.add_resources_used else None
    new_traces = new_trace_start([log]) if label.add_new_traces else None
    kwargs = {'executed_events': executed_events, 'resources_used': resources_used, 'new_traces': new_traces,
              'label': label}
    return kwargs


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
               atr_classifier=None, executed_events=None, resources_used=None, new_traces=None):
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
        labels.append(trace.attributes[atr_classifier])
    elif label.type == DURATION:
        labels.append(duration(trace))
    return labels
