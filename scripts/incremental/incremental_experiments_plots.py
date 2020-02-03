import itertools

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
        '0-40_80-100': 138,
        '0-80_80-100': 139,
        '40-80_80-100': 140,
    },
    'BPI11/f2/': {
        '0-40_80-100': 141,
        '0-80_80-100': 142,
        '40-80_80-100': 143,
    },
    'BPI11/f3/': {
        '0-40_80-100': 144,
        '0-80_80-100': 145,
        '40-80_80-100': 146,
    },
    'BPI11/f4/': {
        '0-40_80-100': 147,
        '0-80_80-100': 148,
        '40-80_80-100': 149,
    },
    'BPI15/f1/': {
        '0-40_80-100': 150,
        '0-80_80-100': 151,
        '40-80_80-100': 152,
    },
    'BPI15/f2/': {
        '0-40_80-100': 153,
        '0-80_80-100': 154,
        '40-80_80-100': 155,
    },
    'BPI15/f3/': {
        '0-40_80-100': 156,
        '0-80_80-100': 157,
        '40-80_80-100': 158,
    },
    'Drift1/f1/': {
        '0-40_80-100': 159,
        '0-80_80-100': 160,
        '40-80_80-100': 161,

        '40-60_80-100': 1111,
        '0-60_80-100': 1111,
        '40-55_80-100': 1111,
        '0-55_80-100': 1111
    },
    'Drift2/f1/': {
        '0-40_80-100': 162,
        '0-80_80-100': 163,
        '40-80_80-100': 164,

        '40-60_80-100': 1111,
        '0-60_80-100': 1111,
        '40-55_80-100': 1111,
        '0-55_80-100': 1111
    }
}
# }
# splits = {
#     'BPI11/f1/': {
#         '0-40_80-100': 101,
#         '0-80_80-100': 102,
#         '40-80_80-100': 103,
#     },
#     'BPI11/f2/': {
#         '0-40_80-100': 104,
#         '0-80_80-100': 105,
#         '40-80_80-100': 106,
#     },
#     'BPI11/f3/': {
#         '0-40_80-100': 107,
#         '0-80_80-100': 108,
#         '40-80_80-100': 109,
#     },
#     'BPI11/f4/': {
#         '0-40_80-100': 110,
#         '0-80_80-100': 111,
#         '40-80_80-100': 112,
#     },
#     'BPI15/f1/': {
#         '0-40_80-100': 113,
#         '0-80_80-100': 114,
#         '40-80_80-100': 115,
#     },
#     'BPI15/f2/': {
#         '0-40_80-100': 116,
#         '0-80_80-100': 117,
#         '40-80_80-100': 118,
#     },
#     'BPI15/f3/': {
#         '0-40_80-100': 119,
#         '0-80_80-100': 120,
#         '40-80_80-100': 121,
#     },
#     'Drift1/f1/': {
#         '0-40_80-100': 122,
#         '0-80_80-100': 123,
#         '40-80_80-100': 124,
#
#         '40-60_80-100': 1111,
#         '0-60_80-100': 1111,
#         '40-55_80-100': 1111,
#         '0-55_80-100': 1111
#     },
#     'Drift2/f1/': {
#         '0-40_80-100': 125,
#         '0-80_80-100': 126,
#         '40-80_80-100': 127,
#
#         '40-60_80-100': 1111,
#         '0-60_80-100': 1111,
#         '40-55_80-100': 1111,
#         '0-55_80-100': 1111
#     }
# }

# splits = {
#     'BPI11/f1/': {
#         '0-40_80-100': 55,
#         '0-80_80-100': 56,
#         '40-80_80-100': 73,
#     },
#     'BPI11/f2/': {
#         '0-40_80-100': 57,
#         '0-80_80-100': 58,
#         '40-80_80-100': 74,
#     },
#     'BPI11/f3/': {
#         '0-40_80-100': 59,
#         '0-80_80-100': 60,
#         '40-80_80-100': 75,
#     },
#     'BPI11/f4/': {
#         '0-40_80-100': 61,
#         '0-80_80-100': 62,
#         '40-80_80-100': 76,
#     },
#     'BPI15/f1/': {
#         '0-40_80-100': 63,
#         '0-80_80-100': 64,
#         '40-80_80-100': 77,
#     },
#     'BPI15/f2/': {
#         '0-40_80-100': 65,
#         '0-80_80-100': 66,
#         '40-80_80-100': 78,
#     },
#     'BPI15/f3/': {
#         '0-40_80-100': 67,
#         '0-80_80-100': 68,
#         '40-80_80-100': 79,
#     },
#     'Drift1/f1/': {
#         '0-40_80-100': 69,
#         '0-80_80-100': 70,
#         '40-80_80-100': 80,
#
#         '40-60_80-100': 1111,
#         '0-60_80-100': 1111,
#         '40-55_80-100': 1111,
#         '0-55_80-100': 1111
#     },
#     'Drift2/f1/': {
#         '0-40_80-100': 90,#71,
#         '0-80_80-100': 91,#72,
#         '40-80_80-100': 92,#81,
#
#         '40-60_80-100': 1111,
#         '0-60_80-100': 1111,
#         '40-55_80-100': 1111,
#         '0-55_80-100': 1111
#     }
# }


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


def quantitative_scores(experiments_df_path='../DUMP_INCREMENTAL_hyper_last_train.csv', where_save='quantitative_scores.csv'):
    pd.set_option("display.precision", 4)
    experiments_df = pd.read_csv(experiments_df_path)

    splits_scores = {}

    for split_id in list(set(experiments_df['split_id'].unique()) & set(splits[dataset]['0-40_80-100'] for dataset in datasets)):
        table = experiments_df[(experiments_df['split_id'] == split_id) &
                               (experiments_df['incremental_model_id'].isnull()) &
                               ((experiments_df['hyperparameter_optimizer_performance_metric'].notnull()) |
                                (experiments_df['hyperparameter_optimizer_max_evaluations'].notnull()) |
                                (experiments_df['hyperparameter_optimizer_elapsed_time'].notnull()))&
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
                                (experiments_df['hyperparameter_optimizer_elapsed_time'].isnull()))]# &
                               # (experiments_df['predictive_model'] != 'randomForest')]  # M1
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
                                (experiments_df['hyperparameter_optimizer_elapsed_time'].notnull()))]# &
                               # (experiments_df['predictive_model'] != 'randomForest')]  # M2
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
                               (experiments_df['hyperparameter_optimizer_elapsed_time'].isnull()))]# &
                                # (experiments_df['predictive_model'] != 'randomForest')]  # M3
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


def create_macro_table(experiments_df_path='../DUMP_INCREMENTAL_hyper_last_train.csv', where_save='macro_table_avg_std_max_NO_rf.csv'):
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
                           experiments_df_path='../DUMP_INCREMENTAL_hyper_last_train.csv',
                           where_save='macro_table_avg_std_NO_rf.csv'):
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
                                   (experiments_df[objective] == objective_id) ]  # M2

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


if __name__ == '__main__':

    #TODO: read in the model object in memory to ease the analysis
    # models = json.load(open("created_jobs.txt"))

    aggregates_dfs = []
    aggregates_dfs += [create_macro_table(experiments_df_path='../DUMP_INCREMENTAL_hyper_last_train.csv', where_save='macro_table_avg_std_max_NO_rf.csv')]
    aggregates_dfs += [
        create_macro_obj_table(
            objective=obj,
            experiments_df_path='../DUMP_INCREMENTAL_hyper_last_train.csv',
            where_save='macro_table_' + obj + '_avg_std_NO_rf.csv')
        for obj in [ 'predictive_model', 'encoding_prefix_length', 'encoding_value_encoding' ]
    ]
    quantitative_scores(experiments_df_path='../DUMP_INCREMENTAL_hyper_last_train.csv', where_save='macro_table_avg_std_max_NO_rf.csv')



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


def avg_std_plot():
    plt.clf()
    plt.hold(1)

    x = np.linspace(0, 30, 30)
    y = np.sin(x / 6 * np.pi)
    error = np.random.normal(0.1, 0.02, size=y.shape) + .1
    y += np.random.normal(0, 0.1, size=y.shape)

    plt.plot(x, y, 'k', color='#CC4F1B')
    plt.fill_between(x, y - error, y + error,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

    y = np.cos(x / 6 * np.pi)
    error = np.random.rand(len(y)) * 0.5
    y += np.random.normal(0, 0.1, size=y.shape)
    plt.plot(x, y, 'k', color='#1B2ACC')
    plt.fill_between(x, y - error, y + error,
                     alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
                     linewidth=4, linestyle='dashdot', antialiased=True)

    y = np.cos(x / 6 * np.pi) + np.sin(x / 3 * np.pi)
    error = np.random.rand(len(y)) * 0.5
    y += np.random.normal(0, 0.1, size=y.shape)
    plt.plot(x, y, 'k', color='#3F7F4C')
    plt.fill_between(x, y - error, y + error,
                     alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',
                     linewidth=0)

    plt.show()
