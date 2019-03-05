import numpy as np
import pandas as pd
from pandas import DataFrame
from pm4py.objects.log.log import Trace, EventLog

from src.encoding.models import Encoding, TaskGenerationTypes, ValueEncodings
from src.encoding.simple_index import compute_label_columns, add_labels, get_intercase_attributes
from src.labelling.models import Labelling


def boolean(log: EventLog, event_names: list, label: Labelling, encoding: Encoding) -> DataFrame:
    return _encode_boolean_frequency(log, event_names, label, encoding)


def frequency(log: EventLog, event_names: list, label: Labelling, encoding: Encoding) -> DataFrame:
    return _encode_boolean_frequency(log, event_names, label, encoding)


def _encode_boolean_frequency(log: EventLog, event_names: list, labelling: Labelling,
                              encoding: Encoding) -> DataFrame:
    """Encodes the log by boolean or frequency

    trace_id, event_nr, event_names, label stuff
    :return pandas DataFrame
    """
    columns = _create_columns(event_names, encoding, labelling)
    encoded_data = []

    kwargs = get_intercase_attributes(log, encoding)
    for trace in log:
        if len(trace) <= encoding.prefix_length - 1 and not encoding.padding:
            # trace too short and no zero padding
            continue
        if encoding.task_generation_type == TaskGenerationTypes.ALL_IN_ONE.value:
            for i in range(1, min(encoding.prefix_length + 1, len(trace) + 1)):
                encoded_data.append(
                    _trace_to_row(trace, encoding, i, labelling, event_names=event_names, atr_classifier=labelling.attribute_name,
                                  **kwargs))
        else:
            encoded_data.append(
                _trace_to_row(trace, encoding, encoding.prefix_length, labelling, event_names=event_names,
                              atr_classifier=labelling.attribute_name, **kwargs))

    return pd.DataFrame(columns=columns, data=encoded_data)


def _create_event_happened(event_names: list, encoding: Encoding) -> list:
    """Creates list of event happened placeholders"""
    if encoding.value_encoding == ValueEncodings.BOOLEAN.value:
        return [False] * len(event_names)
    return [0] * len(event_names)


def _update_event_happened(event, event_names: list, event_happened: list, encoding: Encoding) -> None:
    """Updates the event_happened list at event index

    For boolean set happened to True.
    For frequency updates happened count.
    """
    event_name = event['concept:name']
    if event_name in event_names:
        event_index = event_names.index(event_name)
        if encoding.value_encoding == ValueEncodings.BOOLEAN.value:
            event_happened[event_index] = True
        else:
            event_happened[event_index] += 1


def _create_columns(event_names: list, encoding: Encoding, labelling: Labelling) -> list:
    columns = ["trace_id"]
    columns = list(np.append(columns, event_names).tolist())
    return compute_label_columns(columns, encoding, labelling)


def _trace_to_row(trace: Trace, encoding: Encoding, event_index: int, labelling: Labelling = None, executed_events=None,
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
    trace_row += add_labels(encoding, labelling, event_index, trace, attribute_classifier=atr_classifier,
                            executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
    if trace_row[-1] in event_names:
        trace_row[-1] = event_names.index(trace_row[-1])
    return trace_row
