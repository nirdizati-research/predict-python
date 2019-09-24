import json
import time

import pandas
import requests


def upload_split(train='cache/log_cache/test_logs/general_example_train.xes',
                 test='cache/log_cache/test_logs/general_example_test.xes'):
    SERVER_NAME = "0.0.0.0"
    SERVER_PORT = '8000'
    # SERVER_PORT = '50401'
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
    HYPERPARAMETER_OPTIMIZATION={"type": 'hyperopt',"max_evaluations": 1000,"performance_metric": "auc", "algorithm_type": "tpe"},
    INCREMENTAL_TRAIN={"base_model": None},
    MODEL_HYPERPARAMETERS={}):

    CONFIG = {
        "clusterings": CLUSTERING,
        "labelling": LABELING,
        "encodings": ENCODINGS,
        "encoding": ENCODING,
        "hyperparameter_optimizer": HYPERPARAMETER_OPTIMIZATION,
        "methods": CLASSIFICATION,
        "incremental_train": INCREMENTAL_TRAIN,
        "create_models": "True",
        }
    CONFIG.update(MODEL_HYPERPARAMETERS)

    return {"type": "classification", "split_id": SPLIT, "config": CONFIG}


def send_job_request(PAYLOAD):
    SERVER_NAME = "0.0.0.0"
    SERVER_PORT = '8000'
    # SERVER_PORT = '50401'
    headers = {'Content-type': 'application/json'}
    r = requests.post('http://' + SERVER_NAME + ':' + SERVER_PORT + '/jobs/multiple', json=PAYLOAD, headers=headers)
    return json.loads(r.text)


def init_database(splits, dataset, base_folder):
    splits[dataset] = {}

    splits[dataset]['0-40_80-100'] = upload_split(train=base_folder + dataset + '0-40.xes',
                                                  test=base_folder + dataset + '80-100.xes')

    splits[dataset]['0-60_80-100'] = upload_split(train=base_folder + dataset + '0-60.xes',
                                                  test=base_folder + dataset + '80-100.xes')

    splits[dataset]['0-80_80-100'] = upload_split(train=base_folder + dataset + '0-80.xes',
                                                  test=base_folder + dataset + '80-100.xes')

    splits[dataset]['40-60_80-100'] = upload_split(train=base_folder + dataset + '40-60.xes',
                                                   test=base_folder + dataset + '80-100.xes')

    splits[dataset]['60-80_80-100'] = upload_split(train=base_folder + dataset + '60-80.xes',
                                                   test=base_folder + dataset + '80-100.xes')


def incremental_experiments(dataset, prefix_length, models, splits, classification_method, encoding_method):
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

    models[dataset]['0-60_80-100'] = send_job_request(
        PAYLOAD=create_payload(
            SPLIT=splits[dataset]['0-60_80-100'],
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

    models[dataset]['0-80_80-100'] = send_job_request(
        PAYLOAD=create_payload(
            SPLIT=splits[dataset]['0-80_80-100'],
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

    if classification_method != "randomForest":
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

def get_pretrained_model_id(data, prefix, attribute_name, classification_method, dataset, encoding_method):
    model_id = data.loc[
        (data['predictive_model'] == classification_method) &
        (data['encoding_value_encoding'] == encoding_method) &
        (data['encoding_prefix_length'] == prefix) &
        (data['labelling_attribute_name'] == attribute_name) &
        (data['split_id'] == dataset)
    ].filter(items=['predictive_model_id']).values[0][0]
    return model_id


def launch_std_experimentation(
    datasets,
    splits,
    base_folder,
    models,
    prefixes=[10, 30, 50, 70],
    classification_methods=["multinomialNB", "SGDClassifier", "perceptron", "randomForest"],
    encodings=["complex", "simpleIndex"]
):
    for dataset in datasets:
        init_database(splits, dataset, base_folder)

        print(dataset, '[:::] Batch of logs uploaded')
        models[dataset] = {}

        for prefix_length in prefixes: # NB: if you add something the splits and models are overwritten
            for classification_method in classification_methods: # NB: if you add something the models are overwritten
                for encoding_method in encodings: # NB: if you add something the models are overwritten
                    incremental_experiments( dataset, prefix_length, models, splits, classification_method, encoding_method)

        print(dataset, '[:::] Batch of tasks created')
        time.sleep(180)


def launch_drift_size_experimentation(
    datasets,
    splits,
    base_folder,
    prefixes=[10, 30, 50, 70],
    classification_methods=["multinomialNB", "SGDClassifier", "perceptron", "randomForest"],
    encodings=["complex", "simpleIndex"]
):

    for dataset in datasets:
        splits[dataset]['40-55_80-100'] = upload_split(train=base_folder + dataset + '40-55.xes',
                                                       test=base_folder + dataset + '80-100.xes')
        splits[dataset]['0-55_80-100'] = upload_split(train=base_folder + dataset + '0-55.xes',
                                                      test=base_folder + dataset + '80-100.xes')

        for prefix_length in prefixes:  # NB: if you add something the splits and models are overwritten
            for classification_method in classification_methods:  # NB: if you add something the models are overwritten
                for encoding_method in encodings:  # NB: if you add something the models are overwritten

                    if classification_method != "randomForest":
                        models[dataset]['40-55_80-100'] = send_job_request(
                            PAYLOAD=create_payload(
                                SPLIT=splits[dataset]['40-55_80-100'],
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
                                INCREMENTAL_TRAIN={
                                    "base_model": get_pretrained_model_id(data, prefix_length, 'label', classification_method,
                                                                          splits[dataset]['0-40_80-100'], encoding_method)}
                                # MODEL_HYPERPARAMETERS={
                                #     'classification_' + classification_method: Job.objects.filter(id=models[dataset]['0-40_80-100'])[0].predictive_model.to_dict()
                                # }
                            )
                        )[0]['id']

                    models[dataset]['0-55_80-100'] = send_job_request(
                        PAYLOAD=create_payload(
                            SPLIT=splits[dataset]['0-55_80-100'],
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

        print(dataset, '[:::] Batch of tasks created')
        time.sleep(180)


if __name__ == '__main__':
    print("Starting experiments")

    # base_folder = '/home/wrizzi/Documents/datasets/'
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

    splits = {
        'BPI11/f1/': {
            '0-40_80-100': 1,
            '0-60_80-100': 2,
            '0-80_80-100': 3,
            '40-60_80-100': 4,
            '60-80_80-100': 5
        },
        'BPI11/f2/': {
            '0-40_80-100': 6,
            '0-60_80-100': 7,
            '0-80_80-100': 8,
            '40-60_80-100': 9,
            '60-80_80-100': 10
        },
        'BPI11/f3/': {
            '0-40_80-100': 11,
            '0-60_80-100': 12,
            '0-80_80-100': 13,
            '40-60_80-100': 14,
            '60-80_80-100': 15
        },
        'BPI11/f4/': {
            '0-40_80-100': 16,
            '0-60_80-100': 17,
            '0-80_80-100': 18,
            '40-60_80-100': 19,
            '60-80_80-100': 20
        },
        'BPI15/f1/': {
            '0-40_80-100': 21,
            '0-60_80-100': 22,
            '0-80_80-100': 23,
            '40-60_80-100': 24,
            '60-80_80-100': 25
        },
        'BPI15/f2/': {
            '0-40_80-100': 26,
            '0-60_80-100': 27,
            '0-80_80-100': 28,
            '40-60_80-100': 29,
            '60-80_80-100': 30
        },
        'BPI15/f3/': {
            '0-40_80-100': 31,
            '0-60_80-100': 32,
            '0-80_80-100': 33,
            '40-60_80-100': 34,
            '60-80_80-100': 35
        },
        'Drift1/f1/': {
            '0-40_80-100': 36,
            '0-60_80-100': 37,
            '0-80_80-100': 38,
            '40-60_80-100': 39,
            '60-80_80-100': 40,
            '40-55_80-100': 48,  # 46 #TODO: error in upload of the 46
            '0-55_80-100': 49  # 47  #TODO: error in upload of the 46
        },
        'Drift2/f1/': {
            '0-40_80-100': 41,
            '0-60_80-100': 42,
            '0-80_80-100': 43,
            '40-60_80-100': 44,
            '60-80_80-100': 45,
            '40-55_80-100': 50,  # 48 #TODO: error in upload of the 47
            '0-55_80-100': 51  # 49 #TODO: error in upload of the 47
        }
    }
    models = {}

    data = pandas.read_csv('DUMP_INCREMENTAL.csv', parse_dates=['evaluation_elapsed_time'])

    launch_std_experimentation(
        datasets,
        splits,
        base_folder,
        models,
        prefixes=[10, 30, 50, 70],
        classification_methods=["multinomialNB", "SGDClassifier", "perceptron", "randomForest"],
        encodings=["complex", "simpleIndex"]
    )

    launch_drift_size_experimentation(
        datasets=datasets,
        splits=splits,
        base_folder=base_folder,
        prefixes=[10, 30, 50, 70],
        classification_methods=["multinomialNB", "SGDClassifier", "perceptron", "randomForest"],
        encodings=["complex", "simpleIndex"]
    )

    print("End of the experiemnts")
