"""
regression methods and functionalities
"""
from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn import clone
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from xgboost import XGBRegressor

from src.clustering.clustering import Clustering
from src.core.common import get_method_config
from src.encoding.models import Encoding
from src.jobs.models import Job, ModelType
from src.predictive_model.regression.custom_regression_models import NNRegressor
from src.predictive_model.regression.models import RegressionMethods
from src.utils.django_orm import duplicate_orm_row
from src.utils.result_metrics import _prepare_results

pd.options.mode.chained_assignment = None

import logging

logger = logging.getLogger(__name__)


def regression(training_df: DataFrame, test_df: DataFrame, clusterer: Clustering, job: Job) -> (dict, dict):
    """main regression entry point

    train and tests the regressor using the provided data

    :param clusterer:
    :param training_df: training DataFrame
    :param test_df: testing DataFrame
    :param job: job configuration
    :return: predictive_model scores and split

    """
    train_data, test_data = _prep_data(training_df, test_df)

    # job.encoding = duplicate_orm_row(Encoding.objects.filter(pk=job.encoding.pk)[0])  # TODO: maybe here would be better an intelligent get_or_create...
    job.encoding = Encoding.objects.create(
        data_encoding=job.encoding.data_encoding,
        value_encoding=job.encoding.value_encoding,
        add_elapsed_time=job.encoding.add_elapsed_time,
        add_remaining_time=job.encoding.add_remaining_time,
        add_executed_events=job.encoding.add_executed_events,
        add_resources_used=job.encoding.add_resources_used,
        add_new_traces=job.encoding.add_new_traces,
        features=job.encoding.features,
        prefix_length=job.encoding.prefix_length,
        padding=job.encoding.padding,
        task_generation_type=job.encoding.task_generation_type
    )
    job.encoding.features = list(train_data.columns.values)
    job.encoding.save()
    job.save()

    model_split = _train(train_data, _choose_regressor(job), clusterer)
    results_df = _test(model_split, test_data)

    results = _prepare_results(results_df, job.labelling)

    return results, model_split


def cross_validated_regression(training_df: DataFrame, test_df: DataFrame, clusterer: Clustering, job: Job, cv=2) -> (dict, dict):
    """main regression entry point

    train and tests the regressor using the provided data

    :param clusterer:
    :param training_df: training DataFrame
    :param test_df: testing DataFrame
    :param job: job configuration
    :param cv: cross validation amount
    :return: predictive_model scores and split

    """
    train_data, test_data = _prep_data(training_df, test_df)

    # job.encoding = duplicate_orm_row(Encoding.objects.filter(pk=job.encoding.pk)[0])  # TODO: maybe here would be better an intelligent get_or_create...
    job.encoding = Encoding.objects.create(
        data_encoding=job.encoding.data_encoding,
        value_encoding=job.encoding.value_encoding,
        add_elapsed_time=job.encoding.add_elapsed_time,
        add_remaining_time=job.encoding.add_remaining_time,
        add_executed_events=job.encoding.add_executed_events,
        add_resources_used=job.encoding.add_resources_used,
        add_new_traces=job.encoding.add_new_traces,
        features=job.encoding.features,
        prefix_length=job.encoding.prefix_length,
        padding=job.encoding.padding,
        task_generation_type=job.encoding.task_generation_type
    )
    job.encoding.features = list(train_data.columns.values)
    job.encoding.save()
    job.save()

    model_split = _train(train_data, _choose_regressor(job), clusterer, do_cv=True)
    results_df = _test(model_split, test_data)

    results = calculate_results_regression(results_df, job.labelling)

    return results, model_split


def regression_single_log(input_df: DataFrame, model: dict) -> DataFrame:
    """single log regression

    classifies a single log using the provided TODO: complete

    :param input_df: input DataFrame
    :param model: TODO: complete
    :return: predictive_model scores

    """
    split = model['split']
    input_df = input_df.drop([col for col in ['label', 'remaining_time', 'trace_id'] if col in input_df.columns], 1)

    # TODO load predictive_model more wisely
    model_split = dict()
    model_split[ModelType.CLUSTERER.value] = joblib.load(split['clusterer_path'])
    model_split[ModelType.REGRESSOR.value] = joblib.load(split['model_path'])
    results_df = _test(model_split, input_df)
    return results_df


def _train(train_data: DataFrame, regressor: RegressorMixin, clusterer: Clustering, do_cv=False) -> dict:
    models = dict()

    train_data = clusterer.cluster_data(train_data)

    for cluster in range(clusterer.n_clusters):

        cluster_train_df = train_data[cluster]
        if not cluster_train_df.empty:
            cluster_targets_df = cluster_train_df['label']

            if do_cv:
                cross_validation_result = cross_validate(
                    regressor,
                    cluster_train_df.drop('label', 1),
                    cluster_targets_df.values.ravel(),
                    return_estimator=True,
                    cv=10 #TODO per Chiara check se vuoi 10 cv
                )

                validation_scores = cross_validation_result['test_score']
                regressors = cross_validation_result['estimator']
                regressor = regressors[dict(zip(validation_scores,range(len(validation_scores))))[max(validation_scores)]] #TODO per Chiara check se vuoi il max o min o quello che sta in mezzo
            else:
                regressor.fit(cluster_train_df.drop('label', 1), cluster_targets_df.values.ravel())

            models[cluster] = regressor
            try:
                regressor = clone(regressor)
            except TypeError:
                regressor = clone(regressor, safe=False)

    return {ModelType.CLUSTERER.value: clusterer, ModelType.REGRESSOR.value: models}


def _test(model_split: dict, data: DataFrame) -> DataFrame:
    clusterer = model_split[ModelType.CLUSTERER.value]
    regressor = model_split[ModelType.REGRESSOR.value]

    test_data = clusterer.cluster_data(data)

    results_df = DataFrame()

    for cluster in range(clusterer.n_clusters):
        cluster_test_df = test_data[cluster]
        if not cluster_test_df.empty:
            cluster_test_df['predicted'] = regressor[cluster].predict(cluster_test_df.drop('label', 1))
            results_df = results_df.append(cluster_test_df)
    return results_df


def predict(job: Job, data: DataFrame) -> Any:
    data = data.drop(['trace_id'], 1)
    clusterer = Clustering.load_model(job)
    test_data = clusterer.cluster_data(data)

    regressor = joblib.load(job.predictive_model.model_path)

    result = None

    for cluster in range(clusterer.n_clusters):
        cluster_test_df = test_data[cluster]
        if not cluster_test_df.empty:
            result = regressor[cluster].predict(cluster_test_df.drop('label', 1))

    return result


def _prep_data(training_df: DataFrame, test_df: DataFrame) -> (DataFrame, DataFrame):
    train_data = training_df
    test_data = test_df

    test_data = test_data.drop(['trace_id'], 1)
    train_data = train_data.drop('trace_id', 1)
    return train_data, test_data


def _choose_regressor(job: Job) -> RegressorMixin:
    method, config = get_method_config(job)
    config.pop('regression_method', None)
    logger.info("Using method {} with config {}".format(method, config))
    if method == RegressionMethods.LINEAR.value:
        regressor = LinearRegression(**config)
    elif method == RegressionMethods.RANDOM_FOREST.value:
        regressor = RandomForestRegressor(**config)
    elif method == RegressionMethods.LASSO.value:
        regressor = Lasso(**config)
    elif method == RegressionMethods.XGBOOST.value:
        regressor = XGBRegressor(**config)
    elif method == RegressionMethods.NN.value:
        config['encoding'] = job.encoding.value_encoding
        regressor = NNRegressor(**config)
    else:
        raise ValueError("Unexpected regression method {}".format(method))
    return regressor
