import json

import requests

import pandas as pd
import matplotlib.pyplot as plt


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
    '0-40_80-100',
    '0-60_80-100',
    '0-80_80-100',
    '40-60_80-100',
    '60-80_80-100',
    '40-55_80-100',
    '0-55_80-100'
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

prefixes = [10, 30, 50, 70]
classification_methods = ["multinomialNB", "SGDClassifier", "perceptron", "randomForest"]
encodings = ["complex", "simpleIndex"]
task_generation_type = ['all_in_one']

datasets_by_id = {}
for dataset in datasets:
    for split_size in split_sizes:
        if dataset in splits and split_size in splits[dataset]:
            datasets_by_id[splits[dataset][split_size]] = dataset + split_size


def retrieve_job(ID):
    # SERVER_NAME = "0.0.0.0"
    # SERVER_PORT = '8000'
    SERVER_NAME = "ashkin"
    SERVER_PORT = '50401'
    headers = {'Content-type': 'application/json'}
    r = requests.get('http://' + SERVER_NAME + ':' + SERVER_PORT + '/jobs/' + str(ID), headers=headers)
    return json.loads(r.text)


def get_dataset_by_id(ID):
    if ID in datasets_by_id:
        return datasets_by_id[ID]
    else:
        for dataset in datasets:
            for split_size in split_sizes:
                if dataset in splits and split_size in splits[dataset] and splits[dataset][split_size] == ID:
                    return dataset + split_size

    return str(ID) + ' Not found'


experiments_already_done_list = []

for index in range(1300):
    job_result = retrieve_job(ID=index)
    if not('detail' in job_result and job_result['detail'] == 'Not found.'):
        experiments_already_done_list += [(
                get_dataset_by_id(ID=job_result['config']['split']['id']),
                job_result['config']['encoding']['prefix_length'],
                job_result['config']['encoding']['task_generation_type'],
                job_result['config']['encoding']['value_encoding'],
                job_result['config']['predictive_model']['prediction_method'],
                'incremental' if job_result['config']['incremental_train'] is not None else 'traditional',
                'OK' if job_result['status'] == 'completed' else 'NOK'
        )]

experiments_already_done = pd.DataFrame(experiments_already_done_list, columns=['dataset', 'prefix', 'sampling', 'encoding', 'classifier', 'train_type', 'status'])
experiments_already_done.to_csv('./incremental_experiments.csv')


def plot_unit():
    plt.figure(figsize=(9, 3))

    plt.subplot(131)
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    plt.axis([0, 6, 0, 20])
    plt.xlabel('entry a')
    plt.ylabel('some numbers')

    plt.subplot(132)
    plt.plot([1, 2, 3, 4, 5, 6, 7])
    plt.plot([5, 6, 7])
    plt.plot([1,6,89])
    plt.axis([0, 6, 0, 20])
    plt.xlabel('entry a')
    plt.ylabel('some numbers')

    plt.subplot(133)
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    plt.axis([0, 6, 0, 20])
    plt.xlabel('entry a')
    plt.ylabel('some numbers')

    plt.suptitle('Categorical Plotting')
    plt.show()

    plt.suptitle('Categorical Plotting')
    plt.show()

