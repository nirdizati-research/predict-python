from typing import Callable

import pandas as pd
from pandas import DataFrame
from pm4py.objects.log.log import Trace, EventLog

from src.encoding.encoder import PREFIX_
from src.encoding.models import Encoding, TaskGenerationTypes
from src.encoding.simple_index import compute_label_columns, add_labels, get_intercase_attributes
from src.labelling.models import Labelling

ATTRIBUTE_CLASSIFIER = None


def complex(log: EventLog, labelling: Labelling, encoding: Encoding, additional_columns: dict) -> DataFrame:
    """Encodes the log using complex encoding

    :param log:
    :param labelling:
    :param encoding:
    :param additional_columns:
    :return:
    """
    return _encode_complex_latest(log, labelling, encoding, additional_columns, _columns_complex, _data_complex)


def last_payload(log: EventLog, labelling: Labelling, encoding: Encoding, additional_columns: dict) -> DataFrame:
    """Encodes the log using last_payload encoding

    :param log:
    :param labelling:
    :param encoding:
    :param additional_columns:
    :return:
    """
    return _encode_complex_latest(log, labelling, encoding, additional_columns, _columns_last_payload,
                                  _data_last_payload)


def _encode_complex_latest(log: EventLog, labelling: Labelling, encoding: Encoding, additional_columns: dict,
                           column_fun: Callable, data_fun: Callable) -> DataFrame:
    """Encodes the log by complex or last_payload

    :param log:
    :param labelling:
    :param encoding:
    :param additional_columns:
    :param column_fun:
    :param data_fun:
    :return:
    """
    columns = column_fun(encoding.prefix_length, additional_columns)
    normal_columns_number = len(columns)
    columns = compute_label_columns(columns, encoding, labelling)
    encoded_data = []

    kwargs = get_intercase_attributes(log, encoding)
    for trace in log:
        if len(trace) <= encoding.prefix_length - 1 and not encoding.padding:
            # trace too short and no zero padding
            continue
        if encoding.task_generation_type == TaskGenerationTypes.ALL_IN_ONE.value:
            for i in range(1, min(encoding.prefix_length + 1, len(trace) + 1)):
                encoded_data.append(
                    _trace_to_row(trace, encoding, labelling, i, data_fun, normal_columns_number,
                                  additional_columns=additional_columns,
                                  atr_classifier=labelling.attribute_name, **kwargs))
        else:
            encoded_data.append(
                _trace_to_row(trace, encoding, labelling, encoding.prefix_length, data_fun, normal_columns_number,
                              additional_columns=additional_columns,
                              atr_classifier=labelling.attribute_name, **kwargs))
    return pd.DataFrame(columns=columns, data=encoded_data)


def _columns_complex(prefix_length: int, additional_columns: dict) -> list:
    """Creates list in form [1, value1, value2, 2, ...]

    Appends values in additional_columns
    :param prefix_length:
    :param additional_columns:
    :return:
    """
    columns = ['trace_id']
    columns += additional_columns['trace_attributes']
    for i in range(1, prefix_length + 1):
        columns.append(PREFIX_ + str(i))
        for additional_column in additional_columns['event_attributes']:
            columns.append(additional_column + "_" + str(i))
    return columns


def _columns_last_payload(prefix_length: int, additional_columns: dict) -> list:
    """Creates list in form [1, 2, value1, value2,]

    Appends values in additional_columns
    :param prefix_length:
    :param additional_columns:
    :return:
    """
    columns = ['trace_id']
    i = 0
    for i in range(1, prefix_length + 1):
        columns.append(PREFIX_ + str(i))
    for additional_column in additional_columns['event_attributes']:
        columns.append(additional_column + "_" + str(i))
    return columns


def _data_complex(trace: Trace, prefix_length: int, additional_columns: dict) -> list:
    """Creates list in form [1, value1, value2, 2, ...]

    Appends values in additional_columns
    :param trace:
    :param prefix_length:
    :param additional_columns:
    :return:
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
    :param trace:
    :param prefix_length:
    :param additional_columns:
    :return:
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


def _trace_to_row(trace: Trace, encoding: Encoding, labelling: Labelling, event_index: int, data_fun: Callable,
                  columns_len: int,
                  atr_classifier=None, executed_events=None, resources_used=None, new_traces=None,
                  additional_columns: dict = None) -> list:
    """Transforms trace into a raw

    :param trace:
    :param encoding:
    :param labelling:
    :param event_index:
    :param data_fun:
    :param columns_len:
    :param atr_classifier:
    :param executed_events:
    :param resources_used:
    :param new_traces:
    :param additional_columns
    :return:
    """
    trace_row = [trace.attributes["concept:name"]]
    # prefix_length - 1 == index
    trace_row += data_fun(trace, event_index, additional_columns)
    if encoding.padding or encoding.task_generation_type == TaskGenerationTypes.ALL_IN_ONE.value:
        trace_row += [0 for _ in range(len(trace_row), columns_len)]
    trace_row += add_labels(encoding, labelling, event_index, trace, attribute_classifier=atr_classifier,
                            executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
    return trace_row
