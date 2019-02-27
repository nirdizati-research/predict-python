from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB

from skmultiflow.trees import HoeffdingTree, HAT

# Classification methods
KNN = 'knn'
RANDOM_FOREST = 'randomForest'
DECISION_TREE = 'decisionTree'
MULTINOMIAL_NAIVE_BAYES = MultinomialNB().__class__.__name__
ADAPTIVE_TREE = HAT().__class__.__name__
HOEFFDING_TREE = HoeffdingTree().__class__.__name__
SGDCLASSIFIER = SGDClassifier().__class__.__name__
PERCEPTRON = Perceptron().__class__.__name__
NN = 'nn'

# Regression methods
LINEAR = 'linear'
LASSO = 'lasso'
XGBOOST = 'xgboost'

# Time Series Prediction methods
RNN = 'rnn'

# Clustering
KMEANS = 'kmeans'
DBSCAN = 'dbscan'
NO_CLUSTER = 'noCluster'

# Job types
CLASSIFICATION = 'classification'
REGRESSION = 'regression'
LABELLING = 'labelling'
TIME_SERIES_PREDICTION = 'timeSeriesPrediction'
UPDATE = 'update'

# classification/regressor method confs
CLASSIFICATION_RANDOM_FOREST = '{}.{}'.format(CLASSIFICATION, RANDOM_FOREST)
CLASSIFICATION_KNN = '{}.{}'.format(CLASSIFICATION, KNN)
CLASSIFICATION_DECISION_TREE = '{}.{}'.format(CLASSIFICATION, DECISION_TREE)
CLASSIFICATION_MULTINOMIAL_NAIVE_BAYES = '{}.{}'.format(CLASSIFICATION, MULTINOMIAL_NAIVE_BAYES)
CLASSIFICATION_ADAPTIVE_TREE = '{}.{}'.format(CLASSIFICATION, ADAPTIVE_TREE)
CLASSIFICATION_HOEFFDING_TREE = '{}.{}'.format(CLASSIFICATION, HOEFFDING_TREE)
CLASSIFICATION_SGDC = '{}.{}'.format(CLASSIFICATION, SGDCLASSIFIER)
CLASSIFICATION_PERCEPTRON = '{}.{}'.format(CLASSIFICATION, PERCEPTRON)
CLASSIFICATION_XGBOOST = '{}.{}'.format(CLASSIFICATION, XGBOOST)
CLASSIFICATION_NN = '{}.{}'.format(CLASSIFICATION, NN)

REGRESSION_LASSO = '{}.{}'.format(REGRESSION, LASSO)
REGRESSION_LINEAR = '{}.{}'.format(REGRESSION, LINEAR)
REGRESSION_XGBOOST = '{}.{}'.format(REGRESSION, XGBOOST)
REGRESSION_RANDOM_FOREST = '{}.{}'.format(REGRESSION, RANDOM_FOREST)
REGRESSION_NN = '{}.{}'.format(REGRESSION, NN)

TIME_SERIES_PREDICTION_RNN = '{}.{}'.format(TIME_SERIES_PREDICTION, RNN)

UPDATE_INCREMENTAL_NAIVE_BAYES = '{}.{}'.format(UPDATE, MULTINOMIAL_NAIVE_BAYES)
UPDATE_INCREMENTAL_ADAPTIVE_TREE = '{}.{}'.format(UPDATE, ADAPTIVE_TREE)
UPDATE_INCREMENTAL_HOEFFDING_TREE = '{}.{}'.format(UPDATE, HOEFFDING_TREE)

classification_methods = [KNN, DECISION_TREE, RANDOM_FOREST,
                          XGBOOST, MULTINOMIAL_NAIVE_BAYES, HOEFFDING_TREE,
                          ADAPTIVE_TREE, SGDCLASSIFIER, PERCEPTRON,
                          NN]

regression_methods = [LINEAR, RANDOM_FOREST, LASSO, XGBOOST, NN]

time_series_prediction_methods = [RNN]

all_configs = [
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
    UPDATE_INCREMENTAL_NAIVE_BAYES,
    UPDATE_INCREMENTAL_ADAPTIVE_TREE,
    UPDATE_INCREMENTAL_HOEFFDING_TREE,
    REGRESSION_NN,

    TIME_SERIES_PREDICTION_RNN
]
