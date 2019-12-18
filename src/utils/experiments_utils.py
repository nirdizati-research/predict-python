import json

import requests

from src.clustering.models import ClusteringMethods
from src.encoding.models import ValueEncodings, TaskGenerationTypes
from src.hyperparameter_optimization.models import HyperOptAlgorithms, HyperOptLosses, HyperparameterOptimizationMethods
from src.labelling.models import LabelTypes, ThresholdTypes
from src.predictive_model.classification.models import ClassificationMethods
from src.predictive_model.regression.models import RegressionMethods


def create_classification_payload(
    split=1,
    encodings=[ValueEncodings.SIMPLE_INDEX.value],
    encoding={
        "padding": "zero_padding",
        "generation_type": TaskGenerationTypes.ALL_IN_ONE.value,
        "prefix_length": 5,
        "features": []
    },
    labeling={
        "type": LabelTypes.ATTRIBUTE_STRING.value,
        "attribute_name": "creator",
        "threshold_type": ThresholdTypes.THRESHOLD_MEAN.value,
        "threshold": 0,
        "add_remaining_time": False,
        "add_elapsed_time": False,
        "add_executed_events": False,
        "add_resources_used": False,
        "add_new_traces": False
    },
    clustering=[ClusteringMethods.NO_CLUSTER.value],
    classification=[ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value],
    hyperparameter_optimization={
        "type": HyperparameterOptimizationMethods.HYPEROPT.value,
        "max_evaluations": 3,
        "performance_metric": HyperOptLosses.AUC.value,
        "algorithm_type": HyperOptAlgorithms.TPE.value
    },
    incremental_train=[],
    model_hyperparameters={}):

    config = {
        "clusterings": clustering,
        "labelling": labeling,
        "encodings": encodings,
        "encoding": encoding,
        "hyperparameter_optimizer": hyperparameter_optimization,
        "methods": classification,
        "incremental_train": incremental_train,
        "create_models": True,
        }
    config.update(model_hyperparameters)

    return {"type": "classification", "split_id": split, "config": config}


def create_regression_payload(
    split=1,
    encodings=[ValueEncodings.SIMPLE_INDEX.value],
    encoding={
        "padding": "zero_padding",
        "generation_type": TaskGenerationTypes.ALL_IN_ONE.value,
        "prefix_length": 5,
        "features": []
    },
    labeling={
        "type": LabelTypes.ATTRIBUTE_STRING.value,
        "attribute_name": "creator",
        "threshold_type": ThresholdTypes.THRESHOLD_MEAN.value,
        "threshold": 0,
        "add_remaining_time": False,
        "add_elapsed_time": False,
        "add_executed_events": False,
        "add_resources_used": False,
        "add_new_traces": False
    },
    clustering=[ClusteringMethods.NO_CLUSTER.value],
    regression=[RegressionMethods.RANDOM_FOREST.value],
    hyperparameter_optimization={
        "type": HyperparameterOptimizationMethods.HYPEROPT.value,
        "max_evaluations": 3,
        "performance_metric": HyperOptLosses.RMSE.value,
        "algorithm_type": HyperOptAlgorithms.TPE.value
    },
    incremental_train=[],
    model_hyperparameters={}):

    config = {
        "clusterings": clustering,
        "labelling": labeling,
        "encodings": encodings,
        "encoding": encoding,
        "hyperparameter_optimizer": hyperparameter_optimization,
        "methods": regression,
        "incremental_train": incremental_train,
        "create_models": True,
        }
    config.update(model_hyperparameters)

    return {"type": "regression", "split_id": split, "config": config}


def upload_split(
    train='cache/log_cache/test_logs/general_example_train.xes',
    test='cache/log_cache/test_logs/general_example_test.xes',
    server_name="0.0.0.0",
    server_port='8000'
):
    r = requests.post(
        'http://' + server_name + ':' + server_port + '/splits/multiple',
        files={'trainingSet': open(train, 'r+'), 'testSet': open(test, 'r+')}
    )
    return json.loads(r.text)['id']


def send_job_request(
    payload,
    server_name="0.0.0.0",
    server_port='8000'
):
    r = requests.post(
        'http://' + server_name + ':' + server_port + '/jobs/multiple',
        json=payload,
        headers={'Content-type': 'application/json'}
    )
    return json.loads(r.text)


def retrieve_job(
    config,
    server_name="0.0.0.0",
    server_port='8000'
):
    r = requests.get(
        'http://' + server_name + ':' + server_port + '/jobs/',
        headers={'Content-type': 'application/json'},
        json=config
    )
    return json.loads(r.text)
