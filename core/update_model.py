import copy

from sklearn.externals import joblib

from core.constants import MULTINOMIAL_NAIVE_BAYES, NO_CLUSTER, CLASSIFICATION, ADAPTIVE_TREE, \
    HOEFFDING_TREE
from predModels.models import PredModels
from utils.result_metrics import calculate_results_classification


def retrieve_model_from_cache(job : dict):

    try:
        model = PredModels.objects.filter(type=CLASSIFICATION)[len(PredModels.objects.filter(type=CLASSIFICATION)) - 1]#TODO TO TEST
    except:
        raise KeyError("PredModel not available in DB")

    clf = joblib.load(model.split.model_path)

    return clf, model


def choose_clf_model(job : dict):
    method = job['method']
    if method == MULTINOMIAL_NAIVE_BAYES:
        clf, model = retrieve_model_from_cache(job)
    elif method == ADAPTIVE_TREE:
        clf, model = retrieve_model_from_cache(job)
    elif method == HOEFFDING_TREE:
        clf, model = retrieve_model_from_cache(job)
    else:
        raise ValueError('Unexpected Incremental method %s' % method)
    return clf, model


def update_model(training_df, test_df, job):

    training_df, test_df = add_actual(training_df, test_df)

    train_data, test_data, original_test_data = drop_columns(training_df, test_df)

    if job['clustering'] == NO_CLUSTER:
        clf, model = choose_clf_model(job)

        results_df, auc, model_split, clf = no_clustering_update(original_test_data, train_data, clf)
    # elif job['clustering'] == KMEANS:
    #     results_df, auc, model_split = kmeans_clustering_update(original_test_data, train_data, clf, job['kmeans'])
    else:
        raise ValueError("Unexpected clustering %s" % job['clustering'])

    if model.split.model_path[-5] is not 'n':
        model_split['versioning'] = int(model.split.model_path[-5])
    else:
        model_split[ 'versioning' ] = 0

    results = prepare_results(results_df, auc)

    return results, model_split


def add_actual(training_df, test_df):
    training_df['actual'] = training_df['label']
    test_df['actual'] = test_df['label']
    return training_df, test_df


def drop_columns(training_df, test_df):
    training_df = training_df.drop(['label', 'trace_id'], 1)
    original_test_df = test_df.drop('label', 1)
    test_df = test_df.drop(['label', 'trace_id'], 1)
    return training_df, test_df, original_test_df


def prepare_results(df, auc: int):
    actual_ = df['actual'].values
    predicted_ = df['predicted'].values

    row = calculate_results_classification(actual_, predicted_)
    row['auc'] = auc
    return row

