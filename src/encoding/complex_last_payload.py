from typing import Callable

import pandas as pd
from pandas import DataFrame
from pm4py.objects.log.log import Trace

from src.encoding.encoding_container import EncodingContainer
from src.encoding.simple_index import compute_label_columns, add_labels, get_intercase_attributes
from src.labelling.label_container import LabelContainer

ATTRIBUTE_CLASSIFIER = None


def complex(log: list, label: LabelContainer, encoding: EncodingContainer, additional_columns: dict) -> DataFrame:
    return _encode_complex_latest(log, label, encoding, additional_columns, _columns_complex, _data_complex)


def last_payload(log: list, label: LabelContainer, encoding: EncodingContainer, additional_columns: dict) -> DataFrame:
    return _encode_complex_latest(log, label, encoding, additional_columns, _columns_last_payload, _data_last_payload)


def _encode_complex_latest(log: list, label: LabelContainer, encoding: EncodingContainer, additional_columns: dict,
                           column_fun: Callable, data_fun: Callable) -> DataFrame:
    columns = column_fun(encoding.prefix_length, additional_columns)
    normal_columns_number = len(columns)
    columns = compute_label_columns(columns, label)
    encoded_data = []

    kwargs = get_intercase_attributes(log, label)
    for trace in log:
        if len(trace) <= encoding.prefix_length - 1 and not encoding.is_zero_padding():
            # trace too short and no zero padding
            continue
        if encoding.is_all_in_one():
            for i in range(1, min(encoding.prefix_length + 1, len(trace) + 1)):
                encoded_data.append(
                    _trace_to_row(trace, encoding, i, data_fun, normal_columns_number,
                                  additional_columns=additional_columns,
                                  atr_classifier=label.attribute_name, **kwargs))
        else:
            encoded_data.append(
                _trace_to_row(trace, encoding, encoding.prefix_length, data_fun, normal_columns_number,
                              additional_columns=additional_columns,
                              atr_classifier=label.attribute_name, **kwargs))
    return pd.DataFrame(columns=columns, data=encoded_data)


def _columns_complex(prefix_length: int, additional_columns: dict) -> list:
    columns = ['trace_id']
    columns += additional_columns['trace_attributes']
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
        for additional_column in additional_columns['event_attributes']:
            columns.append(additional_column + "_" + str(i))
    return columns


def _columns_last_payload(prefix_length: int, additional_columns: dict) -> list:
    columns = ['trace_id']
    i = 0
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
    for additional_column in additional_columns['event_attributes']:
        columns.append(additional_column + "_" + str(i))
    return columns


def _data_complex(trace: Trace, prefix_length: int, additional_columns: dict) -> list:
    """Creates list in form [1, value1, value2, 2, ...]

    Appends values in additional_columns
    """
    data = [trace.attributes.get(att, 0) for att in additional_columns['trace_attributes']]
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = event["concept:name"]
        data.append(event_name)

        for att in additional_columns['event_attributes']:
            data.append(event.get(att, '0'))

    return data


def _data_last_payload(trace: list, prefix_length: int, additional_columns: dict) -> list:
    """Creates list in form [1, 2, value1, value2,]

    Event name index of the position they are in event_names
    Appends values in additional_columns
    """
    data = list()
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = event['concept:name']
        data.append(event_name)

    # Attributes of last event
    for att in additional_columns['event_attributes']:
        if prefix_length - 1 >= len(trace):
            value = 0
        else:
            value = trace[prefix_length - 1][att]
        data.append(value)
    return data


def _trace_to_row(trace: Trace, encoding: EncodingContainer, event_index: int, data_fun: Callable, columns_len: int,
                  atr_classifier=None, label=None, executed_events=None, resources_used=None, new_traces=None,
                  additional_columns: dict = None) -> list:
    trace_row = [trace.attributes["concept:name"]]
    # prefix_length - 1 == index
    trace_row += data_fun(trace, event_index, additional_columns)
    if encoding.is_zero_padding() or encoding.is_all_in_one():
        trace_row += [0 for _ in range(len(trace_row), columns_len)]
    trace_row += add_labels(label, event_index, trace, attribute_classifier=atr_classifier,
                            executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
    return trace_row
