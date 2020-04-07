from collections import defaultdict
from time import time
from opyenxes.data_in.XUniversalParser import XUniversalParser
from itertools import groupby

from opyenxes.model import XAttributeBoolean, XAttributeLiteral, XAttributeTimestamp, XAttributeDiscrete, \
    XAttributeContinuous

def get_attribute_type(val):
    if isinstance(val, XAttributeLiteral.XAttributeLiteral):
        return "literal"
    elif isinstance(val, XAttributeBoolean.XAttributeBoolean):
        return "boolean"
    elif isinstance(val, XAttributeDiscrete.XAttributeDiscrete):
        return "discrete"
    elif isinstance(val, XAttributeTimestamp.XAttributeTimestamp):
        return "timestamp"
    elif isinstance(val, XAttributeContinuous.XAttributeContinuous):
        return "continuous"



def split_log_train_test(log, train_size, test_size=None): #TODO JONAS to be removed
    last_ind = (int)(len(log) * train_size)
    return log[:last_ind], log[last_ind:]

def read_XES_log(path):
    tic = time()

    print("Parsing log")
    with open(path) as log_file:
        log = XUniversalParser().parse(log_file)[0]  # take first log from file

    toc = time()

    print("Log parsed, took {} seconds..".format(toc - tic))

    return log


def extract_attributes(event, attribs=None):
    if not attribs:
        #attribs = ["concept:name", "lifecycle:transition"]
        attribs = ["concept:name"]


    extracted = {}
    event_attribs = event.get_attributes()

    for att in attribs:
        extracted[att] = event_attribs[att].get_value()

    return extracted


def xes_to_positional(log, label=True):
    """
    [
        {tracename:name, tracelabel:label,
         events:{event_a : [1,2,3], event_b : [4,5,6], event_c : [7,8,9]} }
    ]

    :param log:
    :return:
    """

    positional = [

    ]
    #
    # for trace in log:
    #     trace_attribs = trace.get_attributes()
    #     trace_name = trace_attribs["concept:name"].get_value()
    #     if label:
    #         trace_label = int(trace_attribs["Label"].get_value())
    #
    #     events = {}
    #     for pos, event in enumerate(trace):
    #         event_attribs = extract_attributes(event)
    #         event_name = event_attribs["concept:name"]
    #         if "lifecycle:transition" in event_attribs:
    #             event_name = event_name + "-" + str(event_attribs["lifecycle:transition"])
    #         if event_name not in events:
    #             events[event_name] = []
    #         # transition? not for now
    #         events[event_name].append(pos)
    #
    #     positional.append({
    #         "name": trace_name,
    #         "events": events
    #     })
    #     if label:
    #         positional[-1]["label"] = trace_label

    return {
        trace.attributes['concept:name']: {
            key: [item[0] for item in group]
            for key, group in groupby(sorted(enumerate([event['concept:name'] for event in trace]), key=lambda x: x[1]), lambda x: x[1])
        }
        for trace in log
    }



def xes_to_data_positional(log, label=True, considered=None):
    """
    [
        {tracename:name, tracelabel:label,
         events:{event_a : [1,2,3], event_b : [4,5,6], event_c : [7,8,9]} },
         data: [{},{}]
    ]

    :param log:
    :return:
    """

    positional = [

    ]

    for trace in log:
        trace_attribs = trace.get_attributes()
        trace_name = trace_attribs["concept:name"].get_value()
        if label:
            trace_label = int(trace_attribs["Label"].get_value())

        trace_data_flow = []
        trace_data = {}

        # TODO: Add considered values!
        for key, val in trace_attribs.items():
            if key not in set(["concept:name", "time:timestamp", "Label", "lifecycle:transition"]):
                ptype = get_attribute_type(val)
                trace_data[(key, ptype)] = str(val)

        first_event = True
        events = {}
        for pos, event in enumerate(trace):
            event_attribs = extract_attributes(event)
            event_name = event_attribs["concept:name"]

            atrbs = event.get_attributes()
            event_data_attribs = {}
            for key, val in atrbs.items():
                if key not in set(["concept:name", "time:timestamp", "Label", "lifecycle:transition"]):
                    ptype = get_attribute_type(val)
                    event_data_attribs[(key, ptype)] = str(val)

            if first_event:
                for k, val in trace_data.items():
                    if k not in event_data_attribs:
                        event_data_attribs[k] = val

                first_event = False

            trace_data_flow.append(event_data_attribs)

            if "lifecycle:transition" in event_attribs:
                event_name = event_name + "-" + str(event_attribs["lifecycle:transition"])
            if event_name not in events:
                events[event_name] = []
            # transition? not for now
            events[event_name].append(pos)

        positional.append({
            "name": trace_name,
            "events": events,
            "data": trace_data_flow
        })
        if label:
            positional[-1]["label"] = trace_label

    return positional



def extract_unique_events(log):
    unique_events = set()
    for trace in log:
        for event in trace:
            event_attribs = trace.get_attributes()
            if "lifecycle:transition" in event_attribs:
                event_name = event_name + "-" + str(event_attribs["lifecycle:transition"])
            unique_events.add(extract_attributes(event)["concept:name"])

    return unique_events


def extract_unique_events_transformed(log):
    unique_events = set()
    for trace in log:
        for key in trace["events"].keys():
            unique_events.add(key)

    return unique_events
