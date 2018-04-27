import pandas as pd
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier

from encoders.label_container import LabelContainer, ATTRIBUTE_STRING, ATTRIBUTE_NUMBER
from encoders.simple_index import add_label_columns, add_labels
from log_util.event_attributes import get_event_attributes

CLASSIFIER = XEventAttributeClassifier("Trace name", ["concept:name"])
ATTRIBUTE_CLASSIFIER = None


def complex(log, event_names, label: LabelContainer, prefix_length=1, zero_padding=False):
    if prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    return encode_complex_latest(log, event_names, label, prefix_length, columns_complex, data_complex,
                                 zero_padding)


def last_payload(log, event_names, label: LabelContainer, prefix_length=1, zero_padding=False):
    if prefix_length < 1:
        raise ValueError("Prefix length must be greater than 1")
    return encode_complex_latest(log, event_names, label, prefix_length, columns_last_payload, data_last_payload,
                                 zero_padding)


def encode_complex_latest(log, event_names: list, label: LabelContainer, prefix_length: int, column_fun, data_fun,
                          zero_padding: bool):
    additional_columns = get_event_attributes(log)
    columns = column_fun(prefix_length, additional_columns, label)
    encoded_data = []

    # Create classifier only once
    if label.type == ATTRIBUTE_STRING or label.type == ATTRIBUTE_NUMBER:
        global ATTRIBUTE_CLASSIFIER
        ATTRIBUTE_CLASSIFIER = XEventAttributeClassifier("Attr class", [label.attribute_name])
    for trace in log:
        if zero_padding:
            zero_count = prefix_length - len(trace)
        elif len(trace) <= prefix_length - 1:
            # no padding, skip this trace
            continue
        trace_row = []
        trace_name = CLASSIFIER.get_class_identity(trace)
        trace_row.append(trace_name)
        # prefix_length - 1 == index
        trace_row += data_fun(trace, event_names, prefix_length, additional_columns)
        if zero_padding:
            trace_row += [0 for _ in range(0, zero_count)]
        trace_row += add_labels(label, prefix_length, trace, event_names, ATTRIBUTE_CLASSIFIER=ATTRIBUTE_CLASSIFIER)
        encoded_data.append(trace_row)

    return pd.DataFrame(columns=columns, data=encoded_data)


def columns_complex(prefix_length: int, additional_columns: list, label: LabelContainer):
    columns = ['trace_id']
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
        for additional_column in additional_columns:
            columns.append(additional_column + "_" + str(i))
    return add_label_columns(columns, label)


def columns_last_payload(prefix_length: int, additional_columns: list, label: LabelContainer):
    columns = ['trace_id']
    for i in range(1, prefix_length + 1):
        columns.append("prefix_" + str(i))
    for additional_column in additional_columns:
        columns.append(additional_column + "_" + str(i))
    return add_label_columns(columns, label)


def data_complex(trace: list, event_names: list, prefix_length: int, additional_columns: list):
    """Creates list in form [1, value1, value2, 2, ...]

    Event name index of the position they are in event_names
    Appends values in additional_columns
    """
    data = list()
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = CLASSIFIER.get_class_identity(event)
        event_id = event_names.index(event_name)
        data.append(event_id + 1)  # prefix

        for att in additional_columns:
            # Basically XEventAttributeClassifier
            value = event.get_attributes().get(att).get_value()
            data.append(value)

    return data


def data_last_payload(trace: list, event_names: list, prefix_length: int, additional_columns: list):
    """Creates list in form [1, 2, value1, value2,]

    Event name index of the position they are in event_names
    Appends values in additional_columns
    """
    data = list()
    for idx, event in enumerate(trace):
        if idx == prefix_length:
            break
        event_name = CLASSIFIER.get_class_identity(event)
        event_id = event_names.index(event_name)
        data.append(event_id + 1)  # prefix
    # Attributes of last event
    for att in additional_columns:
        # Basically XEventAttributeClassifier
        value = trace[prefix_length - 1].get_attributes().get(att).get_value()
        data.append(value)
    return data
