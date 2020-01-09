import json

import requests

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


def get_row_metrics(table):
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

    curr_row = pd.to_timedelta(table['evaluation_elapsed_time'])
    elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min = curr_row.mean(), curr_row.std(), curr_row.max(), curr_row.min()

    return f1_score_mean, f1_score_std, f1_score_max, \
           accuracy_mean, accuracy_std, accuracy_max, \
           precision_mean, precision_std, precision_max, \
           recall_mean, recall_std, recall_max, \
           auc_mean, auc_std, auc_max, \
           elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min


def quantitative_scores(experiments_df_path='../DUMP_INCREMENTAL.csv', where_save='quantitative_scores.csv'):
    pd.set_option("display.precision", 4)
    experiments_df = pd.read_csv(experiments_df_path)

    splits_scores = {}

    for split_id in list(set(experiments_df['split_id'].unique()) & set(splits[dataset]['0-40_80-100'] for dataset in datasets)):
        table = experiments_df[(experiments_df['split_id'] == split_id) &
                               (experiments_df['incremental_model_id'].isnull()) &
                               ((experiments_df['hyperparameter_optimizer_performance_metric'].notnull()) |
                                (experiments_df['hyperparameter_optimizer_max_evaluations'].notnull()) |
                                (experiments_df['hyperparameter_optimizer_elapsed_time'].notnull())) &
                               (experiments_df['predictive_model'] != 'randomForest')]  # M0
        m_dataset = [dataset for dataset in datasets if splits[dataset]['0-40_80-100'] == split_id][0]
        if not table.empty:
            table['model'] = 'M0'
            table['dataset'] = m_dataset
            table['size'] = '0-40_80-100'
            if m_dataset not in splits_scores:
                splits_scores[m_dataset] = table
            else:
                splits_scores[m_dataset] = pd.concat([splits_scores[m_dataset], table])

    for split_id in list(set(experiments_df['split_id'].unique()) & set(splits[dataset]['0-80_80-100'] for dataset in datasets)):
        table = experiments_df[(experiments_df['split_id'] == split_id) &
                               (experiments_df['incremental_model_id'].isnull()) &
                               ((experiments_df['hyperparameter_optimizer_performance_metric'].isnull()) &
                                (experiments_df['hyperparameter_optimizer_max_evaluations'].isnull()) &
                                (experiments_df['hyperparameter_optimizer_elapsed_time'].isnull())) &
                               (experiments_df['predictive_model'] != 'randomForest')]  # M1
        m_dataset = [dataset for dataset in datasets if splits[dataset]['0-80_80-100'] == split_id][0]
        if not table.empty:
            table['model'] = 'M1'
            table['dataset'] = m_dataset
            table['size'] = '0-80_80-100'
            if m_dataset not in splits_scores:
                splits_scores[m_dataset] = table
            else:
                splits_scores[m_dataset] = pd.concat([splits_scores[m_dataset], table])

    for split_id in list(set(experiments_df['split_id'].unique()) & set(splits[dataset]['0-80_80-100'] for dataset in datasets)):
        table = experiments_df[(experiments_df['split_id'] == split_id) &
                               (experiments_df['incremental_model_id'].isnull()) &
                               ((experiments_df['hyperparameter_optimizer_performance_metric'].notnull()) |
                                (experiments_df['hyperparameter_optimizer_max_evaluations'].notnull()) |
                                (experiments_df['hyperparameter_optimizer_elapsed_time'].notnull())) &
                               (experiments_df['predictive_model'] != 'randomForest')]  # M2
        m_dataset = [dataset for dataset in datasets if splits[dataset]['0-80_80-100'] == split_id][0]
        if not table.empty:
            table['model'] = 'M2'
            table['dataset'] = m_dataset
            table['size'] = '0-80_80-100'
            if m_dataset not in splits_scores:
                splits_scores[m_dataset] = table
            else:
                splits_scores[m_dataset] = pd.concat([splits_scores[m_dataset], table])

    for split_id in list(set(experiments_df['split_id'].unique()) & set( splits[dataset]['40-80_80-100'] for dataset in datasets )):
        table = experiments_df[(experiments_df['split_id'] == split_id) &
                               (experiments_df['incremental_model_id'].notnull()) &
                               ((experiments_df['hyperparameter_optimizer_performance_metric'].isnull()) &
                               (experiments_df['hyperparameter_optimizer_max_evaluations'].isnull()) &
                               (experiments_df['hyperparameter_optimizer_elapsed_time'].isnull())) &
                               (experiments_df['predictive_model'] != 'randomForest')]  # M3
        m_dataset = [dataset for dataset in datasets if splits[dataset]['40-80_80-100'] == split_id][0]
        if not table.empty:
            table['model'] = 'M3'
            table['dataset'] = m_dataset
            table['size'] = '40-80_80-100'
            if m_dataset not in splits_scores:
                splits_scores[m_dataset] = table
            else:
                splits_scores[m_dataset] = pd.concat([splits_scores[m_dataset], table])

    quantitative_scores = {}
    for dataset in datasets:
        for encoding, prefix, predictive_model in itertools.product(*[
            splits_scores[dataset]['encoding_value_encoding'].unique(),
            splits_scores[dataset]['encoding_prefix_length'].unique(),
            splits_scores[dataset]['predictive_model'].unique()
        ]):
            table = splits_scores[dataset][
                (splits_scores[dataset]['encoding_value_encoding'] == encoding) &
                (splits_scores[dataset]['encoding_prefix_length'] == prefix) &
                (splits_scores[dataset]['predictive_model'] == predictive_model)
            ]
            if len(table) == 4:
                max_conf = table.loc[table['evaluation_auc'].idxmax()]
                if dataset not in quantitative_scores:
                    quantitative_scores[dataset] = pd.DataFrame([max_conf])
                else:
                    quantitative_scores[dataset] = pd.concat([quantitative_scores[dataset], pd.DataFrame([max_conf])])

    if where_save is not None:
        for dataset in datasets:
            splits_scores[dataset].sort_values(by=['evaluation_auc'], inplace=True, ascending=False)
            splits_scores[dataset].to_csv(dataset.replace('/', '_') + '_' + where_save)

            quantitative_scores[dataset].to_csv(dataset.replace('/', '_') + 'models' + '_' + where_save)
    # return aggregates_df


def create_macro_table(experiments_df_path='../DUMP_INCREMENTAL.csv', where_save='macro_table_avg_std_max.csv'):
    pd.set_option("display.precision", 4)
    experiments_df = pd.read_csv(experiments_df_path)
    aggregates_list = []
    for split_id in list(set(experiments_df['split_id'].unique()) & set( splits[dataset]['0-40_80-100'] for dataset in datasets )):
        table = experiments_df[(experiments_df['split_id'] == split_id) &
                               (experiments_df['incremental_model_id'].isnull()) &
                               ((experiments_df['hyperparameter_optimizer_performance_metric'].notnull()) |
                               (experiments_df['hyperparameter_optimizer_max_evaluations'].notnull()) |
                               (experiments_df['hyperparameter_optimizer_elapsed_time'].notnull())) &
                               (experiments_df['predictive_model'] != 'randomForest')]  # M0

        f1_score_mean, f1_score_std, f1_score_max, \
        accuracy_mean, accuracy_std, accuracy_max, \
        precision_mean, precision_std, precision_max, \
        recall_mean, recall_std, recall_max, \
        auc_mean, auc_std, auc_max, \
        elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min = get_row_metrics(table)

        m_dataset = [dataset for dataset in datasets if splits[dataset]['0-40_80-100'] == split_id][0]

        if not table.empty:
            aggregates_list += [[
                m_dataset, '0-40_80-100', 'M0',
                split_id, f1_score_max, accuracy_max, precision_max, recall_max, auc_max,
                elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min,
                f1_score_mean, f1_score_std,
                accuracy_mean, accuracy_std,
                precision_mean, precision_std,
                recall_mean, recall_std,
                auc_mean, auc_std
            ]]

    for split_id in list(set(experiments_df['split_id'].unique()) & set( splits[dataset]['40-80_80-100'] for dataset in datasets )):
        table = experiments_df[(experiments_df['split_id'] == split_id) &
                               (experiments_df['incremental_model_id'].notnull()) &
                               ((experiments_df['hyperparameter_optimizer_performance_metric'].isnull()) &
                               (experiments_df['hyperparameter_optimizer_max_evaluations'].isnull()) &
                               (experiments_df['hyperparameter_optimizer_elapsed_time'].isnull())) &
                               (experiments_df['predictive_model'] != 'randomForest')]  # M3

        f1_score_mean, f1_score_std, f1_score_max, \
        accuracy_mean, accuracy_std, accuracy_max, \
        precision_mean, precision_std, precision_max, \
        recall_mean, recall_std, recall_max, \
        auc_mean, auc_std, auc_max, \
        elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min = get_row_metrics(table)

        m_dataset = [dataset for dataset in datasets if splits[dataset]['40-80_80-100'] == split_id][0]

        custom_split_id = splits[m_dataset]['0-80_80-100']

        if not table.empty:
            aggregates_list += [[
                m_dataset, '40-80_80-100', 'M3',
                custom_split_id + .3, f1_score_max, accuracy_max, precision_max, recall_max, auc_max,
                elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min,
                f1_score_mean, f1_score_std,
                accuracy_mean, accuracy_std,
                precision_mean, precision_std,
                recall_mean, recall_std,
                auc_mean, auc_std
            ]]

    for split_id in list(set(experiments_df['split_id'].unique()) & set( splits[dataset]['0-80_80-100'] for dataset in datasets )):
        table = experiments_df[(experiments_df['split_id'] == split_id) &
                               (experiments_df['incremental_model_id'].isnull()) &
                               ((experiments_df['hyperparameter_optimizer_performance_metric'].isnull()) &
                                (experiments_df['hyperparameter_optimizer_max_evaluations'].isnull()) &
                                (experiments_df['hyperparameter_optimizer_elapsed_time'].isnull())) &
                               (experiments_df['predictive_model'] != 'randomForest')]  # M1

        f1_score_mean, f1_score_std, f1_score_max, \
        accuracy_mean, accuracy_std, accuracy_max, \
        precision_mean, precision_std, precision_max, \
        recall_mean, recall_std, recall_max, \
        auc_mean, auc_std, auc_max, \
        elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min = get_row_metrics(table)

        m_dataset = [dataset for dataset in datasets if splits[dataset]['0-80_80-100'] == split_id][0]

        if not table.empty:
            aggregates_list += [[
                m_dataset, '0-80_80-100', 'M1',
                split_id + .1, f1_score_max, accuracy_max, precision_max, recall_max, auc_max,
                elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min,
                f1_score_mean, f1_score_std,
                accuracy_mean, accuracy_std,
                precision_mean, precision_std,
                recall_mean, recall_std,
                auc_mean, auc_std
            ]]

    for split_id in list(set(experiments_df['split_id'].unique()) & set(splits[dataset]['0-80_80-100'] for dataset in datasets)):
        table = experiments_df[(experiments_df['split_id'] == split_id) &
                               (experiments_df['incremental_model_id'].isnull()) &
                               ((experiments_df['hyperparameter_optimizer_performance_metric'].notnull()) |
                               (experiments_df['hyperparameter_optimizer_max_evaluations'].notnull()) |
                               (experiments_df['hyperparameter_optimizer_elapsed_time'].notnull())) &
                               (experiments_df['predictive_model'] != 'randomForest')]  # M2

        f1_score_mean, f1_score_std, f1_score_max, \
        accuracy_mean, accuracy_std, accuracy_max, \
        precision_mean, precision_std, precision_max, \
        recall_mean, recall_std, recall_max, \
        auc_mean, auc_std, auc_max, \
        elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min = get_row_metrics(table)

        m_dataset = [dataset for dataset in datasets if splits[dataset]['0-80_80-100'] == split_id][0]

        if not table.empty:
            aggregates_list += [[
                m_dataset, '0-80_80-100', 'M2',
                split_id + .2, f1_score_max, accuracy_max, precision_max, recall_max, auc_max,
                elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min,
                f1_score_mean, f1_score_std,
                accuracy_mean, accuracy_std,
                precision_mean, precision_std,
                recall_mean, recall_std,
                auc_mean, auc_std
            ]]

    aggregates_df = pd.DataFrame(aggregates_list, columns=[
        'dataset', 'size', 'model',
        'split_id', 'f1_score_max', 'accuracy_max', 'precision_max', 'recall_max', 'auc_max',
        'elapsed_time_mean', 'elapsed_time_std', 'elapsed_time_max', 'elapsed_time_min',
        'f1_score_mean', 'f1_score_std',
        'accuracy_mean', 'accuracy_std',
        'precision_mean', 'precision_std',
        'recall_mean', 'recall_std',
        'auc_mean', 'auc_std'
    ])

    aggregates_df.sort_values(by=['split_id'], inplace=True)
    if where_save is not None:
        aggregates_df.to_csv(where_save)
    return aggregates_df


def create_macro_obj_table(objective,
                           experiments_df_path='../DUMP_INCREMENTAL.csv',
                           where_save='macro_table_avg_std.csv'):
    pd.set_option("display.precision", 4)
    experiments_df = pd.read_csv(experiments_df_path)
    aggregates_list = []
    for split_id in list(set(experiments_df['split_id'].unique()) & set( splits[dataset]['0-40_80-100'] for dataset in datasets )):
        for objective_id in experiments_df[experiments_df['split_id'] == split_id][objective].unique():
            table = experiments_df[(experiments_df['split_id'] == split_id) &
                                   (experiments_df['incremental_model_id'].isnull()) &
                                   ((experiments_df['hyperparameter_optimizer_performance_metric'].notnull()) |
                                    (experiments_df['hyperparameter_optimizer_max_evaluations'].notnull()) |
                                    (experiments_df['hyperparameter_optimizer_elapsed_time'].notnull())) &
                                   (experiments_df[objective] == objective_id) &
                                   (experiments_df['predictive_model'] != 'randomForest')]  # M0

            f1_score_mean, f1_score_std, f1_score_max, \
            accuracy_mean, accuracy_std, accuracy_max, \
            precision_mean, precision_std, precision_max, \
            recall_mean, recall_std, recall_max, \
            auc_mean, auc_std, auc_max, \
            elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min = get_row_metrics(table)

            m_dataset = [dataset for dataset in datasets if splits[dataset]['0-40_80-100'] == split_id][0]

            if not table.empty:
                aggregates_list += [[
                    m_dataset, '0-40_80-100', 'M0',
                    split_id, objective_id,
                    f1_score_mean, f1_score_std,
                    accuracy_mean, accuracy_std,
                    precision_mean, precision_std,
                    recall_mean, recall_std,
                    auc_mean, auc_std,
                    elapsed_time_mean, elapsed_time_std
                ]]
    for split_id in list(set(experiments_df['split_id'].unique()) & set( splits[dataset]['40-80_80-100'] for dataset in datasets )):
        for objective_id in experiments_df[experiments_df['split_id'] == split_id][objective].unique():
            table = experiments_df[(experiments_df['split_id'] == split_id) &
                                   (experiments_df['incremental_model_id'].notnull()) &
                                   ((experiments_df['hyperparameter_optimizer_performance_metric'].isnull()) &
                                    (experiments_df['hyperparameter_optimizer_max_evaluations'].isnull()) &
                                    (experiments_df['hyperparameter_optimizer_elapsed_time'].isnull())) &
                                   (experiments_df[objective] == objective_id) &
                                   (experiments_df['predictive_model'] != 'randomForest')]  # M3

            f1_score_mean, f1_score_std, f1_score_max, \
            accuracy_mean, accuracy_std, accuracy_max, \
            precision_mean, precision_std, precision_max, \
            recall_mean, recall_std, recall_max, \
            auc_mean, auc_std, auc_max, \
            elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min = get_row_metrics(table)

            m_dataset = [dataset for dataset in datasets if splits[dataset]['40-80_80-100'] == split_id][0]

            custom_split_id = splits[m_dataset]['0-80_80-100']

            if not table.empty:
                aggregates_list += [[
                    m_dataset, '40-80_80-100', 'M3',
                    custom_split_id + .3, objective_id,
                    f1_score_mean, f1_score_std,
                    accuracy_mean, accuracy_std,
                    precision_mean, precision_std,
                    recall_mean, recall_std,
                    auc_mean, auc_std,
                    elapsed_time_mean, elapsed_time_std
                ]]
    for split_id in list(set(experiments_df['split_id'].unique()) & set( splits[dataset]['0-80_80-100'] for dataset in datasets )):
        for objective_id in experiments_df[experiments_df['split_id'] == split_id][objective].unique():
            table = experiments_df[(experiments_df['split_id'] == split_id) &
                                   (experiments_df['incremental_model_id'].isnull()) &
                                   ((experiments_df['hyperparameter_optimizer_performance_metric'].isnull()) &
                                    (experiments_df['hyperparameter_optimizer_max_evaluations'].isnull()) &
                                    (experiments_df['hyperparameter_optimizer_elapsed_time'].isnull())) &
                                   (experiments_df[objective] == objective_id) &
                                   (experiments_df['predictive_model'] != 'randomForest')]  # M1

            f1_score_mean, f1_score_std, f1_score_max, \
            accuracy_mean, accuracy_std, accuracy_max, \
            precision_mean, precision_std, precision_max, \
            recall_mean, recall_std, recall_max, \
            auc_mean, auc_std, auc_max, \
            elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min = get_row_metrics(table)

            m_dataset = [dataset for dataset in datasets if splits[dataset]['0-80_80-100'] == split_id][0]

            if not table.empty:
                aggregates_list += [[
                    m_dataset, '0-80_80-100', 'M1',
                    split_id + .1, objective_id,
                    f1_score_mean, f1_score_std,
                    accuracy_mean, accuracy_std,
                    precision_mean, precision_std,
                    recall_mean, recall_std,
                    auc_mean, auc_std,
                    elapsed_time_mean, elapsed_time_std
                ]]
    for split_id in list(set(experiments_df['split_id'].unique()) & set( splits[dataset]['0-80_80-100'] for dataset in datasets )):
        for objective_id in experiments_df[experiments_df['split_id'] == split_id][objective].unique():
            table = experiments_df[(experiments_df['split_id'] == split_id) &
                                   (experiments_df['incremental_model_id'].isnull()) &
                                   ((experiments_df['hyperparameter_optimizer_performance_metric'].notnull()) |
                                    (experiments_df['hyperparameter_optimizer_max_evaluations'].notnull()) |
                                    (experiments_df['hyperparameter_optimizer_elapsed_time'].notnull())) &
                                   (experiments_df[objective] == objective_id) &
                                   (experiments_df['predictive_model'] != 'randomForest')]  # M2

            f1_score_mean, f1_score_std, f1_score_max, \
            accuracy_mean, accuracy_std, accuracy_max, \
            precision_mean, precision_std, precision_max, \
            recall_mean, recall_std, recall_max, \
            auc_mean, auc_std, auc_max, \
            elapsed_time_mean, elapsed_time_std, elapsed_time_max, elapsed_time_min = get_row_metrics(table)

            m_dataset = [dataset for dataset in datasets if splits[dataset]['0-80_80-100'] == split_id][0]

            if not table.empty:
                aggregates_list += [[
                    m_dataset, '0-80_80-100', 'M2',
                    split_id + .2, objective_id,
                    f1_score_mean, f1_score_std,
                    accuracy_mean, accuracy_std,
                    precision_mean, precision_std,
                    recall_mean, recall_std,
                    auc_mean, auc_std,
                    elapsed_time_mean, elapsed_time_std
                ]]
    aggregates_df = pd.DataFrame(aggregates_list, columns=[
        'dataset', 'size', 'model',
        'split_id', objective,
        'f1_score_mean',  'f1_score_std',
        'accuracy_mean',  'accuracy_std',
        'precision_mean', 'precision_std',
        'recall_mean',    'recall_std',
        'auc_mean',       'auc_std',
        'elapsed_time_mean', 'elapsed_time_std'
    ])

    aggregates_df.sort_values(by=['split_id', objective], inplace=True)
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
    quantitative_scores(experiments_df_path='../DUMP_INCREMENTAL.csv', where_save='macro_table_avg_std_max.csv')



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

