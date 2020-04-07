import pandas as pd
from ast import literal_eval as make_tuple

from pm4py.objects.log.log import Trace

from src.encoding.declare.declaremining import filter_candidates_by_support, generate_train_candidate_constraints, transform_results_to_numpy, reencode_declare_results
from src.encoding.declare.declaretemplates import template_sizes
from src.encoding.declare.declarecommon import xes_to_positional


def declare_deviance_mining(log, labelling, encoding, additional_columns, cols=None): #TODO JONAS
    filter_t = True
    print("Filter_t", filter_t)
    templates = template_sizes.keys()

    constraint_threshold = 0.1
    candidate_threshold = 0.1

    #apply prefix
    log = [Trace(trace[:encoding.prefix_length], attributes=trace.attributes) for trace in log]

    # Read into suitable data structure
    transformed_log = xes_to_positional(log)
    labels = {trace.attributes['concept:name']: trace.attributes['label'] for trace in log}

    # Extract unique activities from log
    events_set = {event_label for tid in transformed_log for event_label in transformed_log[tid]}

    # Brute force all possible candidates
    if cols is None:
        candidates = [(event,) for event in events_set] + [(e1, e2) for e1 in events_set for e2 in events_set if e1 != e2]
    else:
        candidates = list({
            make_tuple(c.split(':')[1]) if len(c.split(':')) > 1 else c
            for c in cols
            if c not in ['label', 'trace_id']
        })
    print("Start candidates:", len(candidates))

    # Count by class
    true_count = len([trace.attributes['concept:name'] for trace in log if trace.attributes['label'] == 'true'])
    false_count = len(log) - true_count
    print("{} deviant and {} normal traces in set".format(false_count, true_count))
    ev_support_true = int(true_count * candidate_threshold)
    ev_support_false = int(false_count * candidate_threshold)

    if filter_t and cols is None:
        print(filter_t)
        print("Filtering candidates by support")
        candidates = filter_candidates_by_support(candidates, transformed_log, labels, ev_support_true, ev_support_false)
        print("Support filtered candidates:", len(candidates))

    constraint_support_false = int(false_count * constraint_threshold)
    constraint_support_true = int(true_count * constraint_threshold)

    train_results = generate_train_candidate_constraints(candidates, templates, transformed_log, labels, constraint_support_true, constraint_support_false, filter_t=filter_t)

    print("Candidate constraints generated")

    # transform to numpy
    # get trace names
    data, labels, featurenames, train_names = transform_results_to_numpy(train_results, labels, transformed_log, cols)

    df = pd.DataFrame(data, columns=featurenames)

    # Reencoding data into one-hot encoding
    # if reencode:
    #     print("Reencoding data")
    #     df, _ = reencode_declare_results(df) #TODO JONAS to be used with only one df

    df["trace_id"] = train_names
    df["label"] = labels.tolist()

    return df
