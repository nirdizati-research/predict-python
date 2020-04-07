"""
Main file for deviance mining
"""
from random import shuffle
import numpy as np
from src.encoding.declare.declaretemplates import *
from src.encoding.declare.declarecommon import *
import pandas as pd

import shutil
import os


def reencode_map(val):
    if val == -1:
        return "violation"
    elif val == 0:
        return "vacuous"
    elif val == 1:
        return "single"
    elif val == 2:
        return "multi"


def reencode_declare_results(df):
    """
    Given declare results dataframe, reencode the results such that they are one-hot encodable
    If Frequency is -1, it means that there was a violation, therefore it will be one class
    If Frequency is 0, it means that the constraint was vacuously filled, it will be second class
    If Frequency is 1, then it will be class of single activation
    If Frequency is 2... then it will be a class of multiple activation

    In total there will be 4 classes
    :param df:
    :return:
    """

    df_size = len(df)

    # First, change all where > 2 to 2.
    df[df > 2] = 2
    # All -1's to "VIOLATION"
    df.replace({
        -1: "violation",
        0: "vacuous",
        1: "single",
        2: "multi"
    }, inplace=True)

    df = pd.get_dummies(data=df, columns=df.columns)
    # Put together and get_dummies for one-encoded features

    df = df.iloc[:df_size, :]

    return df


def apply_template_to_log(template, candidate, log):
    results = []
    for trace in log:
        result, vacuity = apply_template(template, log[trace], candidate)

        results.append(result)

    return results


def generate_candidate_constraints(candidates, templates, train_log, constraint_support=None):
    all_results = {}

    for template in templates:
        print("Started working on {}".format(template))
        for candidate in candidates:
            if len(candidate) == template_sizes[template]:
                constraint_result = apply_template_to_log(template, candidate, train_log)

                if constraint_support:
                    satisfaction_count = len([v for v in constraint_result if v != 0])
                    if satisfaction_count >= constraint_support:
                        all_results[template + ":" + str(candidate)] = constraint_result

                else:
                    all_results[template + ":" + str(candidate)] = constraint_result

    return all_results


def find_if_satisfied_by_class(constraint_result, transformed_log, labels, support_true, support_false):
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


def generate_test_candidate_constraints(candidates, templates, test_log, train_results):
    all_results = {}
    for template in templates:
        print("Started working on {}".format(template))
        for candidate in candidates:
            if len(candidate) == template_sizes[template]:
                candidate_name = template + ":" + str(candidate)
                if candidate_name in train_results:
                    constraint_result = apply_template_to_log(template, candidate, test_log)

                    all_results[candidate_name] = constraint_result

    return all_results


def transform_results_to_numpy(results, labels, transformed_log, cols):
    """
    Transforms results structure into numpy arrays
    :param results:
    :param transformed_log:
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


def count_classes(log):
    false = 0
    true = 0
    for trace in log:
        if trace["label"] == 1:
            false += 1
        else:
            true += 1

    return true, false



def declare_deviance_mining(log, templates=None, to_shuffle=False, filter_t=True, reencode=False):
    print("Filter_t", filter_t)
    if not templates:
        templates = template_sizes.keys()

    constraint_threshold = 0.1
    candidate_threshold = 0.1

    # Read into suitable data structure
    transformed_log = xes_to_positional(log)
    if to_shuffle:
        shuffle(transformed_log)

    train_log, test_log = split_log_train_test(transformed_log, 0.8)

    # Extract unique activities from log
    events_set = extract_unique_events_transformed(train_log)

    # Brute force all possible candidates
    candidates = [(event,) for event in events_set] + [(e1, e2) for e1 in events_set for e2 in events_set if e1 != e2]
    print("Start candidates:", len(candidates))

    # Count by class
    normal_count, deviant_count = count_classes(train_log)
    print("{} deviant and {} normal traces in train set".format(deviant_count, normal_count))
    ev_support_norm = int(normal_count * candidate_threshold)
    ev_support_dev = int(deviant_count * candidate_threshold)

    if filter_t:
        print(filter_t)
        print("Filtering candidates by support")
        candidates = filter_candidates_by_support(candidates, train_log, ev_support_norm, ev_support_dev)
        print("Support filtered candidates:", len(candidates))

    constraint_support_dev = int(deviant_count * constraint_threshold)
    constraint_support_norm = int(normal_count * constraint_threshold)

    train_results = generate_train_candidate_constraints(candidates, templates, train_log, constraint_support_norm,
                                                         constraint_support_dev, filter_t=filter_t)

    test_results = generate_test_candidate_constraints(candidates, templates, test_log, train_results)
    print("Candidate constraints generated")

    # transform to numpy
    # get trace names
    train_data, train_labels, featurenames, train_names = transform_results_to_numpy(train_results, train_log)
    test_data, test_labels, _, test_names = transform_results_to_numpy(test_results, test_log)

    train_df = pd.DataFrame(train_data, columns=featurenames)
    test_df = pd.DataFrame(test_data, columns=featurenames)

    # Reencoding data
    if reencode:
        print("Reencoding data")
        train_df, test_df = reencode_declare_results(train_df, test_df)

    train_df["Case_ID"] = train_names
    train_df["Label"] = train_labels.tolist()
    test_df["Case_ID"] = test_names
    test_df["Label"] = test_labels.tolist()
    train_df.to_csv("declareOutput/declare_train.csv", index=False)
    test_df.to_csv("declareOutput/declare_test.csv", index=False)


def run_deviance_new(log_path, results_folder, templates=None, filter_t=True, reencode=False): #TODO gotta do this
    for logNr in range(5):
        args = {
            "logPath": log_path.format(logNr + 1),
            "labelled": True
        }

        folder_name ="./declareOutput/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        print("Deviance mining filtering:", filter_t)


        deviance_main(args, templates=templates, filter_t=filter_t, reencode=reencode)

        move_out_files_new(logNr + 1, results_folder)


def move_out_files_new(splitNr, results_folder):
    source = './declareOutput/'
    dest1 = './' + results_folder + '/split' + str(splitNr) + "/declare/"

    files = os.listdir(source)

    for f in files:
        shutil.move(source + f, dest1)


def deviance_main(args, templates=None, filter_t=True, reencode=False):
    print("Working on: " + args["logPath"], "Filtering:", filter_t)
    log = read_XES_log(args["logPath"])
    declare_deviance_mining(log, templates=templates, filter_t=filter_t, reencode=reencode)


if __name__ == "__main__":
    log_path = "logs/sepsis_tagged_er.xes"
    args = {
        "logPath": log_path,
        "labelled": True
    }
    deviance_main(args, reencode=True)
