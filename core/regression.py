import pandas as pd
from pandas import DataFrame
from sklearn import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from core.clustering import Clustering
from core.common import get_method_config
from core.constants import LINEAR, RANDOM_FOREST, LASSO, XGBOOST
from utils.result_metrics import calculate_results_regression

pd.options.mode.chained_assignment = None


def regression(training_df: DataFrame, test_df:DataFrame, job: dict):
    train_data, test_data, original_test_data = _prep_data(training_df, test_df)

    model_split = _train(job, train_data, _choose_regressor(job))
    results_df = _test(model_split, test_data)

    results = calculate_results_regression(results_df, job['label'])

    #TODO save model more wisely
    model_split['type'] = job['clustering']

    return results, model_split


def regression_single_log(data: DataFrame, model):
    split = model['split']
    data = data.drop([ col for col in ['label', 'remaining_time', 'trace_id'] if col in data.columns ], 1)

    # TODO load model more wisely
    model_split = dict()
    model_split['clusterer'] = joblib.load(split['clusterer_path'])
    model_split['regressor'] = joblib.load(split['model_path'])
    results_df = _test(model_split, data)
    return results_df


def _train(job: dict, train_data: DataFrame, regressor) -> dict:
    clusterer = Clustering(job)
    models = dict()

    clusterer.fit(train_data.drop('label', 1))

    train_data = clusterer.cluster_data(train_data)

    for cluster in range(clusterer.n_clusters):

        x = train_data[cluster]
        if not x.empty:
            y = x['label']
            regressor.fit(x.drop('label', 1), y)

            models[cluster] = regressor
            try:
                regressor = clone(regressor)
            except TypeError:
                regressor = clone(regressor, safe=False)

    return {'clusterer': clusterer, 'regressor': models}


def _test(model_split, data: DataFrame) -> dict:
    clusterer = model_split['clusterer']
    regressor = model_split['regressor']

    test_data = clusterer.cluster_data(data)

    results_df = DataFrame()

    for cluster in range(clusterer.n_clusters):
        x = test_data[cluster]
        if not x.empty:
            x['predicted'] = regressor[cluster].predict(x.drop('label', 1))
            results_df = results_df.append(x)
    return results_df


def _prep_data(training_df: DataFrame, test_df: DataFrame):
    train_data = training_df
    test_data = test_df

    original_test_data = test_data

    test_data = test_data.drop(['trace_id'], 1)
    train_data = train_data.drop('trace_id', 1)
    return train_data, test_data, original_test_data


def _choose_regressor(job: dict):
    method, config = get_method_config(job)
    print("Using method {} with config {}".format(method, config))
    regressor = None
    if method == LINEAR:
        regressor = LinearRegression(**config)
    elif method == RANDOM_FOREST:
        regressor = RandomForestRegressor(**config)
    elif method == LASSO:
        regressor = Lasso(**config)
    elif method == XGBOOST:
        regressor = XGBRegressor(**config)
    return regressor
