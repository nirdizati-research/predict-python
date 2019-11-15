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
        '0-40_80-100': 8,
        '0-80_80-100': 9,
        '40-80_80-100': 38,
    },
    'BPI11/f2/': {
        '0-40_80-100': 10,
        '0-80_80-100': 11,
        '40-80_80-100': 39,
    },
    'BPI11/f3/': {
        '0-40_80-100': 12,
        '0-80_80-100': 13,
        '40-80_80-100': 40,
    },
    'BPI11/f4/': {
        '0-40_80-100': 14,
        '0-80_80-100': 15,
        '40-80_80-100': 41,
    },
    'BPI15/f1/': {
        '0-40_80-100': 16,
        '0-80_80-100': 17,
        '40-80_80-100': 42,
    },
    'BPI15/f2/': {
        '0-40_80-100': 18,
        '0-80_80-100': 19,
        '40-80_80-100': 43,
    },
    'BPI15/f3/': {
        '0-40_80-100': 20,
        '0-80_80-100': 21,
        '40-80_80-100': 44,
    },
    'Drift1/f1/': {
        '0-40_80-100': 22,
        '0-80_80-100': 23,
        '40-80_80-100': 45,

        '40-60_80-100': 1111,
        '0-60_80-100': 1111,
        '40-55_80-100': 36,  # +TANTO perche' uno e' stato ciccato
        '0-55_80-100': 1111
    },
    'Drift2/f1/': {
        '0-40_80-100': 24,
        '0-80_80-100': 25,
        '40-80_80-100': 46,

        '40-60_80-100': 1111,
        '0-60_80-100': 1111,
        '40-55_80-100': 1111,
        '0-55_80-100': 1111
    }
}

prefixes = [30, 50, 70]
classification_methods = ["multinomialNB", "SGDClassifier", "perceptron", "randomForest"]
encodings = ["complex", "simpleIndex"]
task_generation_type = ['all_in_one']


def compute_done_stuff():
    datasets_by_id = {}
    for dataset in datasets:
        for split_size in split_sizes:
            if dataset in splits and split_size in splits[dataset]:
                datasets_by_id[splits[dataset][split_size]] = dataset + split_size

    def retrieve_job(ID):
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


def create_macro_table(experiments_df_path='../DUMP_INCREMENTAL.csv', where_save='macro_table_avg_std_max.csv'):
    pd.set_option("display.precision", 4)
    experiments_df = pd.read_csv(experiments_df_path)
    aggregates_list = []
    for split_id in experiments_df['split_id'].unique():
        table = experiments_df[experiments_df['split_id'] == split_id]

        curr_row = table['evaluation_f1_score']
        f1_score_mean, f1_score_std, f1_score_max = curr_row.mean(), curr_row.std(), curr_row.max()

        curr_row = table['evaluation_accuracy']
        accuracy_mean, accuracy_std, accuracy_max = curr_row.mean(), curr_row.std(), curr_row.max()

        curr_row = table['evaluation_precision']
        precision_mean, precision_std, precision_max = curr_row.mean(), curr_row.std(), curr_row.max()

        curr_row = table['evaluation_recall']
        recall_mean, recall_std, recall_max = curr_row.mean(), curr_row.std(), curr_row.max()

        curr_row = table['evaluation_auc']
        auc_mean, auc_std, auc_max = curr_row.mean(), curr_row.std(), curr_row.max()

        aggregates_list += [[
            split_id,
            f1_score_mean, f1_score_std, f1_score_max,
            accuracy_mean, accuracy_std, accuracy_max,
            precision_mean, precision_std, precision_max,
            recall_mean, recall_std, recall_max,
            auc_mean, auc_std, auc_max
        ]]
    aggregates_df = pd.DataFrame(aggregates_list, columns=[
        'split_id',
        'f1_score_mean',  'f1_score_std',  'f1_score_max',
        'accuracy_mean',  'accuracy_std',  'accuracy_max',
        'precision_mean', 'precision_std', 'precision_max',
        'recall_mean',    'recall_std',    'recall_max',
        'auc_mean',       'auc_std',       'auc_max'
    ])
    if where_save is not None:
        aggregates_df.to_csv(where_save)
    return aggregates_df


def create_macro_obj_table(objective,
                           experiments_df_path='../DUMP_INCREMENTAL.csv',
                           where_save='macro_table_avg_std.csv'):
    pd.set_option("display.precision", 4)
    experiments_df = pd.read_csv(experiments_df_path)
    aggregates_list = []
    for split_id in experiments_df['split_id'].unique():
        for objective_id in experiments_df[experiments_df['split_id'] == split_id][objective].unique():
            table = experiments_df[(experiments_df['split_id'] == split_id) & (experiments_df[objective] == objective_id)]

            curr_row = table['evaluation_f1_score']
            f1_score_mean, f1_score_std = curr_row.mean(), curr_row.std()

            curr_row = table['evaluation_accuracy']
            accuracy_mean, accuracy_std = curr_row.mean(), curr_row.std()

            curr_row = table['evaluation_precision']
            precision_mean, precision_std = curr_row.mean(), curr_row.std()

            curr_row = table['evaluation_recall']
            recall_mean, recall_std = curr_row.mean(), curr_row.std()

            curr_row = table['evaluation_auc']
            auc_mean, auc_std = curr_row.mean(), curr_row.std()

            aggregates_list += [[
                split_id, objective_id,
                f1_score_mean, f1_score_std,
                accuracy_mean, accuracy_std,
                precision_mean, precision_std,
                recall_mean, recall_std,
                auc_mean, auc_std
            ]]
    aggregates_df = pd.DataFrame(aggregates_list, columns=[
        'split_id', objective,
        'f1_score_mean',  'f1_score_std',
        'accuracy_mean',  'accuracy_std',
        'precision_mean', 'precision_std',
        'recall_mean',    'recall_std',
        'auc_mean',       'auc_std'
    ])
    if where_save is not None:
        aggregates_df.to_csv(where_save)
    return aggregates_df


def compute_summary_table(aggregates_df):
    filtered_aggregates_df = [
        table[
            (table['split_id'] == 1) |
            (table['split_id'] == 3) |
            (table['split_id'] == 5)
        ] for table in aggregates_df
    ]

    d = {}

    # d[1] =
    # d[2] =
    # d[3] =

    index = [
        tuple(x)
        for key, val in [
            (1, 'predictive_model'),
            (2, 'encoding_prefix_length'),
            (3, 'encoding_value_encoding')]
        for x in filtered_aggregates_df[key].get(['split_id', val]).values
    ]

    populations = [
        tuple(x)
        for key, val in [
            (1, 'predictive_model'),
            (2, 'encoding_prefix_length'),
            (3, 'encoding_value_encoding')]
        for x in filtered_aggregates_df[key].drop(['split_id', val], axis=1).values
    ]

    pop = pd.Series(populations, index=index)

    midx = pd.MultiIndex.from_tuples(index)

    pop = pop.reindex(midx)




if __name__ == '__main__':

    #TODO: read in the model object in memory to ease the analysis
    # models = json.load(open("created_jobs.txt"))


    aggregates_dfs = []
    aggregates_dfs += [create_macro_table(experiments_df_path='../DUMP_INCREMENTAL.csv', where_save='macro_table_avg_std_max.csv')]
    aggregates_dfs += [
        create_macro_obj_table(
            objective=obj,
            experiments_df_path='../DUMP_INCREMENTAL.csv',
            where_save='macro_table_' + obj + '_avg_std.csv')
        for obj in [ 'predictive_model', 'encoding_prefix_length', 'encoding_value_encoding' ]
    ]
    # compute_summary_table(aggregates_dfs)


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

