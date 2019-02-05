import numpy as np
import pandas as pd

from encoders.encoding_container import EncodingContainer
from encoders.label_container import LabelContainer
from encoders.simple_index import compute_label_columns, add_labels, get_intercase_attributes


def boolean(log: list, event_names: list, label: LabelContainer, encoding: EncodingContainer):
    return _encode_boolean_frequency(log, event_names, label, encoding)


def frequency(log: list, event_names: list, label: LabelContainer, encoding: EncodingContainer):
    return _encode_boolean_frequency(log, event_names, label, encoding)


def _encode_boolean_frequency(log: list, event_names: list, label: LabelContainer, encoding: EncodingContainer):
    """Encodes the log by boolean or frequency

    trace_id, event_nr, event_names, label stuff
    :return pandas dataframe
    """
    columns = _create_columns(event_names, label)
    encoded_data = []

    kwargs = get_intercase_attributes(log, label)
    for trace in log:
        if len(trace) <= encoding.prefix_length - 1 and not encoding.is_zero_padding():
            # trace too short and no zero padding
            continue
        if encoding.is_all_in_one():
            for i in range(1, min(encoding.prefix_length + 1, len(trace) + 1)):
                encoded_data.append(
                    _trace_to_row(trace, encoding, i, event_names=event_names, atr_classifier=label.attribute_name,
                                  **kwargs))
        else:
            encoded_data.append(
                _trace_to_row(trace, encoding, encoding.prefix_length, event_names=event_names,
                              atr_classifier=label.attribute_name, **kwargs))
    return pd.DataFrame(columns=columns, data=encoded_data)


def _create_event_happened(event_names: list, encoding: EncodingContainer):
    """Creates list of event happened placeholders"""
    if encoding.is_boolean():
        return [False] * len(event_names)
    return [0] * len(event_names)


def _update_event_happened(event, event_names: list, event_happened: list, encoding: EncodingContainer):
    """Updates the event_happened list at event index

    For boolean set happened to True.
    For frequency updates happened count.
    """
    event_name = event['concept:name']
    if event_name in event_names:
        event_index = event_names.index(event_name)
        if encoding.is_boolean():
            event_happened[event_index] = True
        else:
            event_happened[event_index] += 1


def _create_columns(event_names: list, label: LabelContainer):
    columns = ["trace_id"]
    columns = np.append(columns, event_names).tolist()
    return compute_label_columns(columns, label)


def _trace_to_row(trace, encoding: EncodingContainer, event_index: int, label=None, executed_events=None,
                  resources_used=None, new_traces=None, event_names=None, atr_classifier=None):
    # starts with all False, changes to event
    event_happened = _create_event_happened(event_names, encoding)
    trace_row = []
    trace_name = trace.attributes['concept:name']
    trace_row.append(trace_name)
    for index, event in enumerate(trace):
        if index >= event_index:
            pass
        else:
            _update_event_happened(event, event_names, event_happened, encoding)
    trace_row += event_happened
    trace_row += add_labels(label, event_index, trace, atr_classifier=atr_classifier,
                            executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
    return trace_row
