"""
Main file for deviance mining
"""
import numpy as np

from src.encoding.declare.declare_templates import *


def apply_template_to_log(template, candidate, log):
    """returns the log with template applied

    :param template:
    :param candidate:
    :param log:
    :return:
    """
    results = []
    for trace in log:
        result, vacuity = apply_template(template, log[trace], candidate)

        results.append(result)

    return results


def find_if_satisfied_by_class(constraint_result, transformed_log, labels, support_true, support_false):
    """returns two boolean variable show if class is trusted

    :param constraint_result:
    :param transformed_log:
    :param labels:
    :param support_true:
    :param support_false:
    :return:
    """
    fulfill_true = 0
    fulfill_false = 0
    for i, trace in enumerate(transformed_log):
        ## TODO: Find if it is better to have > 0 or != 0.
        if constraint_result[i] > 0:
        #if constraint_result[i] != 0:
            if labels[trace] == 'false':
                fulfill_false += 1
            else:
                fulfill_true += 1

    true_pass = fulfill_true >= support_true
    false_pass = fulfill_false >= support_false

    return true_pass, false_pass


def generate_train_candidate_constraints(candidates, templates, transformed_log, labels, constraint_support_true,
                                         constraint_support_false, filter_t=True):
    """returns the train-candidate's constraints

    :param candidates:
    :param templates:
    :param transformed_log:
    :param labels:
    :param constraint_support_true:
    :param constraint_support_false:
    :param filter_t:
    :return:
    """
    all_results = {}
    for template in templates:
        print("Started working on {}".format(template))
        for candidate in candidates:
            if len(candidate) == template_sizes[template]:
                candidate_name = template + ":" + str(candidate)
                constraint_result = apply_template_to_log(template, candidate, transformed_log)
                satis_true, satis_false = find_if_satisfied_by_class(constraint_result, transformed_log, labels,
                                                                     constraint_support_true,
                                                                     constraint_support_false)

                if not filter_t or (satis_true or satis_false):
                    all_results[candidate_name] = constraint_result

    return all_results


def transform_results_to_numpy(results, labels, transformed_log, cols):
    """
    Transforms results structure into numpy arrays
    :param results:
    :param labels:
    :param transformed_log:
    :param cols:
    :return:
    """
    labels = [labels[trace] for trace in transformed_log]
    trace_names = [trace for trace in transformed_log]
    matrix = []
    featurenames = []

    if cols is None:
        for feature, result in results.items():
            matrix.append(result)
            featurenames.append(feature)
    else:
        for c in cols:
            if c not in ['trace_id', 'label']:
                if c in results:
                    matrix.append(results[c])
                else:
                    matrix.append([0 for _ in range(len(transformed_log))])
                featurenames.append(c)

    nparray_data = np.array(matrix).T
    nparray_labels = np.array(labels)
    nparray_names = np.array(trace_names)
    return nparray_data, nparray_labels, featurenames, nparray_names


def filter_candidates_by_support(candidates, transformed_log, labels, support_true, support_false): #TODO JONAS, no idea what this does
    """returns candidates filtered using given support_true and support_false

    :param candidates:
    :param transformed_log:
    :param labels:
    :param support_true:
    :param support_false:
    :return:
    """
    filtered_candidates = []
    for candidate in candidates:
        count_false = 0
        count_true = 0
        for trace in transformed_log:
            ev_ct = 0
            for event in candidate:
                if event in [event for event in transformed_log[trace]]:
                    ev_ct += 1
                else:
                    break
            if ev_ct == len(candidate):  # all candidate events in trace
                if labels[trace] == 'false':
                    count_false += 1
                else:
                    count_true += 1

            if count_false >= support_false or count_true >= support_true:
                filtered_candidates.append(candidate)
                break

    return filtered_candidates

