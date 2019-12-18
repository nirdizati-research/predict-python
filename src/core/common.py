"""
common methods used in the core package
"""
from src.jobs.models import Job
from src.predictive_model.classification.methods_default_config import classification_random_forest, classification_knn, \
    classification_decision_tree, classification_xgboost, classification_incremental_adaptive_tree, \
    classification_incremental_hoeffding_tree, classification_incremental_sgd_classifier, \
    classification_incremental_perceptron, classification_nn, _update_incremental_naive_bayes, \
    _update_incremental_adaptive_tree, _update_incremental_hoeffding_tree, classification_incremental_naive_bayes
from src.predictive_model.classification.models import CLASSIFICATION_RANDOM_FOREST, CLASSIFICATION_KNN, \
    CLASSIFICATION_DECISION_TREE, CLASSIFICATION_XGBOOST, CLASSIFICATION_MULTINOMIAL_NAIVE_BAYES, \
    CLASSIFICATION_ADAPTIVE_TREE, CLASSIFICATION_HOEFFDING_TREE, CLASSIFICATION_SGDC, CLASSIFICATION_PERCEPTRON, \
    CLASSIFICATION_NN, UPDATE_INCREMENTAL_NAIVE_BAYES, UPDATE_INCREMENTAL_ADAPTIVE_TREE, \
    UPDATE_INCREMENTAL_HOEFFDING_TREE
from src.predictive_model.models import PredictiveModel
from src.predictive_model.regression.methods_default_config import regression_random_forest, regression_xgboost, \
    regression_lasso, regression_linear, regression_nn
from src.predictive_model.regression.models import REGRESSION_RANDOM_FOREST, REGRESSION_XGBOOST, REGRESSION_LASSO, \
    REGRESSION_LINEAR, REGRESSION_NN
from src.predictive_model.time_series_prediction.methods_default_config import time_series_prediction_rnn
from src.predictive_model.time_series_prediction.models import TIME_SERIES_PREDICTION_RNN


def get_method_config(job: Job) -> (str, dict):
    """returns the method configuration dictionary

    :param job: job configuration
    :return: method string and method configuration dict

    """
    method = PredictiveModel.objects.filter(pk=job.predictive_model.pk).select_subclasses()[0]
    config = method.get_full_dict()  # pretty cash money method https://i.imgur.com/vKam04R.png
    config.pop('model_path')
    config.pop('predictive_model')
    method = config['prediction_method']
    config.pop('prediction_method')
    return method, config


ALL_CONFIGS = [
    CLASSIFICATION_RANDOM_FOREST,
    CLASSIFICATION_KNN,
    CLASSIFICATION_DECISION_TREE,
    CLASSIFICATION_XGBOOST,
    CLASSIFICATION_MULTINOMIAL_NAIVE_BAYES,
    CLASSIFICATION_ADAPTIVE_TREE,
    CLASSIFICATION_HOEFFDING_TREE,
    CLASSIFICATION_SGDC,
    CLASSIFICATION_PERCEPTRON,
    CLASSIFICATION_NN,

    REGRESSION_RANDOM_FOREST,
    REGRESSION_LASSO,
    REGRESSION_LINEAR,
    REGRESSION_XGBOOST,
    REGRESSION_NN,

    TIME_SERIES_PREDICTION_RNN,

    UPDATE_INCREMENTAL_NAIVE_BAYES,
    UPDATE_INCREMENTAL_ADAPTIVE_TREE,
    UPDATE_INCREMENTAL_HOEFFDING_TREE
]

CONF_MAP = {
    CLASSIFICATION_RANDOM_FOREST: classification_random_forest,
    CLASSIFICATION_KNN: classification_knn,
    CLASSIFICATION_DECISION_TREE: classification_decision_tree,
    CLASSIFICATION_XGBOOST: classification_xgboost,
    CLASSIFICATION_MULTINOMIAL_NAIVE_BAYES: classification_incremental_naive_bayes,
    CLASSIFICATION_ADAPTIVE_TREE: classification_incremental_adaptive_tree,
    CLASSIFICATION_HOEFFDING_TREE: classification_incremental_hoeffding_tree,
    CLASSIFICATION_SGDC: classification_incremental_sgd_classifier,
    CLASSIFICATION_PERCEPTRON: classification_incremental_perceptron,
    CLASSIFICATION_NN: classification_nn,

    REGRESSION_RANDOM_FOREST: regression_random_forest,
    REGRESSION_XGBOOST: regression_xgboost,
    REGRESSION_LASSO: regression_lasso,
    REGRESSION_LINEAR: regression_linear,
    REGRESSION_NN: regression_nn,

    TIME_SERIES_PREDICTION_RNN: time_series_prediction_rnn,

    UPDATE_INCREMENTAL_NAIVE_BAYES: _update_incremental_naive_bayes,
    UPDATE_INCREMENTAL_ADAPTIVE_TREE: _update_incremental_adaptive_tree,
    UPDATE_INCREMENTAL_HOEFFDING_TREE: _update_incremental_hoeffding_tree
}
