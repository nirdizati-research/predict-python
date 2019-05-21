import json

import requests

print("Starting experiments")


def upload_split(train='cache/log_cache/test_logs/general_example_train.xes',
                 test='cache/log_cache/test_logs/general_example_test.xes'):
    SERVER_NAME = "localhost"
    SERVER_PORT = '8000'
    files = {'trainingSet': open(train, 'r+'), 'testSet': open(test, 'r+')}

    r = requests.post('http://' + SERVER_NAME + ':' + SERVER_PORT + '/splits/multiple', files=files)
    return json.loads(r.text)['id']


def create_payload(
    SPLIT=1,
    ENCODINGS=["simpleIndex"],
    ENCODING={"padding": "zero_padding","generation_type": "all_in_one","prefix_length": 5, "features": []},
    LABELING={"type": "attribute_string","attribute_name": "creator","threshold_type": "threshold_mean","threshold": 0,"add_remaining_time": False,"add_elapsed_time": False,"add_executed_events": False,"add_resources_used": False,"add_new_traces": False},
    CLUSTERING=["noCluster"],
    CLASSIFICATION=["multinomialNB"],
    HYPERPARAMETER_OPTIMIZATION={"type": 'hyperopt',"max_evaluations": 10,"performance_metric": "auc"},
    INCREMENTAL_TRAIN={"base_model": None}):

    CONFIG = {
        "clusterings": CLUSTERING,
        "labelling": LABELING,
        "encodings": ENCODINGS,
        "encoding": ENCODING,
        "hyperparameter_optimizer": HYPERPARAMETER_OPTIMIZATION,
        "methods": CLASSIFICATION,
        "incremental_train": INCREMENTAL_TRAIN,
        "create_models": "True",
        "kmeans": {},
        "classification.knn": {},
        "classification.randomForest": {},
        "classification.decisionTree": {},
        "classification.xgboost": {},
        "classification.multinomialNB": {},
        "classification.hoeffdingTree": {},
        "classification.adaptiveTree": {},
        "classification.SGDClassifier": {},
        "classification.perceptron": {},
        "classification.nn": {},
        "regression.randomForest": {},
        "regression.lasso": {},
        "regression.linear": {},
        "regression.xgboost": {},
        "regression.nn": {},
        "timeSeriesPrediction.rnn": {}}

    return {"type": "classification", "split_id": SPLIT, "config": CONFIG}


def send_job_request(PAYLOAD):
    SERVER_NAME = "localhost"
    SERVER_PORT = '8000'
    headers = {'Content-type': 'application/json'}
    r = requests.post('http://' + SERVER_NAME + ':' + SERVER_PORT + '/jobs/multiple', json=PAYLOAD, headers=headers)
    return json.loads(r.text)


base_folder = '/Users/Brisingr/Desktop/TEMP/dataset/prom_labeled_data/CAiSE18/'

datasets = [
    'BPI11/f1/',
    'BPI11/f2/',
    'BPI11/f3/',
    'BPI11/f4/',

    'BPI15/f1/',
    'BPI15/f2/',
    'BPI15/f3/',

    'Drift1/f1/',

    'Drift2/f1/'
]

split_sizes = [
    '0-40.xes',
    '0-60.xes'
    '0-80.xes'
    '40-60.xes'
    '60-80.xes'
    '80-100.xes'
]

splits = {}
models = {}

for dataset in datasets:
    splits[dataset] = {}

    splits[dataset]['0-40_80-100'] = upload_split(train=base_folder + dataset + '0-40.xes',
                                                  test=base_folder + dataset + '80-100.xes')

    # splits[dataset]['0-60_80-100'] = upload_split(train=base_folder + dataset + '0-60.xes',
    #                                               test=base_folder + dataset + '80-100.xes')
    #
    # splits[dataset]['0-80_80-100'] = upload_split(train=base_folder + dataset + '0-80.xes',
    #                                               test=base_folder + dataset + '80-100.xes')

    splits[dataset]['40-60_80-100'] = upload_split(train=base_folder + dataset + '40-60.xes',
                                                   test=base_folder + dataset + '80-100.xes')

    splits[dataset]['60-80_80-100'] = upload_split(train=base_folder + dataset + '60-80.xes',
                                                   test=base_folder + dataset + '80-100.xes')

    models[dataset] = {}

    for prefix_length in [10]:#, 30, 50]: #NB: if you add something the splits and models are overwritten
        for classification_method in ["multinomialNB"]:#, "SGDClassifier", "perceptron"]: #NB: if you add something the models are overwritten
            for encoding_method in ["complex"]:#, "simpleIndex"]: #NB: if you add something the models are overwritten

                models[dataset]['0-40_80-100'] = send_job_request(
                    PAYLOAD=create_payload(
                        SPLIT=splits[dataset]['0-40_80-100'],
                        ENCODINGS=[encoding_method],
                        ENCODING={"padding": "zero_padding",
                                  "generation_type": "all_in_one",
                                  "prefix_length": prefix_length,
                                  "features": []},
                        LABELING={"type": "attribute_string",
                                  "attribute_name": "label",
                                  "threshold_type": "threshold_mean",
                                  "threshold": 0,
                                  "add_remaining_time": False,
                                  "add_elapsed_time": False,
                                  "add_executed_events": False,
                                  "add_resources_used": False,
                                  "add_new_traces": False},
                        CLASSIFICATION=[classification_method]
                    )
                )[0]['id']


                # models[dataset]['0-60_80-100'] = send_job_request(
                #     PAYLOAD=create_payload(
                #         SPLIT=splits[dataset]['0-60_80-100'],
                #         ENCODINGS=[encoding_method],
                #         ENCODING={"padding": "zero_padding",
                #                   "generation_type": "all_in_one",
                #                   "prefix_length": prefix_length,
                #                   "features": []},
                #         LABELING={"type": "attribute_string",
                #                   "attribute_name": "label",
                #                   "threshold_type": "threshold_mean",
                #                   "threshold": 0,
                #                   "add_remaining_time": False,
                #                   "add_elapsed_time": False,
                #                   "add_executed_events": False,
                #                   "add_resources_used": False,
                #                   "add_new_traces": False},
                #         CLASSIFICATION=[classification_method]
                #     )
                # )[0]['id']
                #
                #
                # models[dataset]['0-80_80-100'] = send_job_request(
                #     PAYLOAD=create_payload(
                #         SPLIT=splits[dataset]['0-80_80-100'],
                #         ENCODINGS=[encoding_method],
                #         ENCODING={"padding": "zero_padding",
                #                   "generation_type": "all_in_one",
                #                   "prefix_length": prefix_length,
                #                   "features": []},
                #         LABELING={"type": "attribute_string",
                #                   "attribute_name": "label",
                #                   "threshold_type": "threshold_mean",
                #                   "threshold": 0,
                #                   "add_remaining_time": False,
                #                   "add_elapsed_time": False,
                #                   "add_executed_events": False,
                #                   "add_resources_used": False,
                #                   "add_new_traces": False},
                #         CLASSIFICATION=[classification_method]
                #     )
                # )[0]['id']


                models[dataset]['40-60_80-100'] = send_job_request(
                    PAYLOAD=create_payload(
                        SPLIT=splits[dataset]['40-60_80-100'],
                        ENCODINGS=[encoding_method],
                        ENCODING={"padding": "zero_padding",
                                  "generation_type": "all_in_one",
                                  "prefix_length": prefix_length,
                                  "features": []},
                        LABELING={"type": "attribute_string",
                                  "attribute_name": "label",
                                  "threshold_type": "threshold_mean",
                                  "threshold": 0,
                                  "add_remaining_time": False,
                                  "add_elapsed_time": False,
                                  "add_executed_events": False,
                                  "add_resources_used": False,
                                  "add_new_traces": False},
                        CLASSIFICATION=[classification_method],
                        HYPERPARAMETER_OPTIMIZATION={"type": 'none'},
                        INCREMENTAL_TRAIN={"base_model": models[dataset]['0-40_80-100']}
                    )
                )[0]['id']


                models[dataset]['60-80_80-100'] = send_job_request(
                    PAYLOAD=create_payload(
                        SPLIT=splits[dataset]['60-80_80-100'],
                        ENCODINGS=[encoding_method],
                        ENCODING={"padding": "zero_padding",
                                  "generation_type": "all_in_one",
                                  "prefix_length": prefix_length,
                                  "features": []},
                        LABELING={"type": "attribute_string",
                                  "attribute_name": "label",
                                  "threshold_type": "threshold_mean",
                                  "threshold": 0,
                                  "add_remaining_time": False,
                                  "add_elapsed_time": False,
                                  "add_executed_events": False,
                                  "add_resources_used": False,
                                  "add_new_traces": False},
                        CLASSIFICATION=[classification_method],
                        HYPERPARAMETER_OPTIMIZATION={"type": 'none'},
                        INCREMENTAL_TRAIN={"base_model": models[dataset]['40-60_80-100']}
                    )
                )[0]['id']


    # json_data = send_job_request(PAYLOAD=create_payload(SPLIT=dataset))
    #
    # INCREMENTAL_TRAIN = {"base_model": json_data[0]['id']}
    # HYPERPARAMETER_OPTIMIZATION = {"type": 'none'}
    #
    # json_data = send_job_request(PAYLOAD=create_payload(SPLIT=1, HYPERPARAMETER_OPTIMIZATION=HYPERPARAMETER_OPTIMIZATION, INCREMENTAL_TRAIN=INCREMENTAL_TRAIN))

print("End of the experiemnts")

