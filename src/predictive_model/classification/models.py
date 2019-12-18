from enum import Enum

from django.db import models

from src.jobs.models import JobTypes
from src.predictive_model.classification.methods_default_config import classification_knn, classification_random_forest, \
    classification_decision_tree, classification_xgboost, classification_incremental_naive_bayes, \
    classification_incremental_hoeffding_tree, classification_incremental_adaptive_tree, \
    classification_incremental_sgd_classifier, classification_incremental_perceptron, classification_nn
from src.predictive_model.models import PredictiveModel, PredictiveModels


class ClassificationMethods(Enum):
    KNN = 'knn'
    RANDOM_FOREST = 'randomForest'
    XGBOOST = 'xgboost'
    DECISION_TREE = 'decisionTree'
    MULTINOMIAL_NAIVE_BAYES = 'multinomialNB'
    ADAPTIVE_TREE = 'adaptiveTree'
    HOEFFDING_TREE = 'hoeffdingTree'
    SGDCLASSIFIER = 'SGDClassifier'
    PERCEPTRON = 'perceptron'
    NN = 'nn'


CLASSIFICATION_METHOD_MAPPINGS = (
    (ClassificationMethods.KNN.value, 'knn'),
    (ClassificationMethods.RANDOM_FOREST.value, 'randomForest'),
    (ClassificationMethods.XGBOOST.value, 'xgboost'),
    (ClassificationMethods.DECISION_TREE.value, 'decisionTree'),
    (ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value, 'multinomialNB'),
    (ClassificationMethods.ADAPTIVE_TREE.value, 'adaptiveTree'),
    (ClassificationMethods.HOEFFDING_TREE.value, 'hoeffdingTree'),
    (ClassificationMethods.SGDCLASSIFIER.value, 'SGDClassifier'),
    (ClassificationMethods.PERCEPTRON.value, 'perceptron'),
    (ClassificationMethods.NN.value, 'nn')
)

CLASSIFICATION_RANDOM_FOREST = '{}.{}'.format(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.RANDOM_FOREST.value)
CLASSIFICATION_KNN = '{}.{}'.format(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.KNN.value)
CLASSIFICATION_DECISION_TREE = '{}.{}'.format(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.DECISION_TREE.value)
CLASSIFICATION_MULTINOMIAL_NAIVE_BAYES = '{}.{}'.format(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value)
CLASSIFICATION_ADAPTIVE_TREE = '{}.{}'.format(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.ADAPTIVE_TREE.value)
CLASSIFICATION_HOEFFDING_TREE = '{}.{}'.format(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.HOEFFDING_TREE.value)
CLASSIFICATION_SGDC = '{}.{}'.format(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.SGDCLASSIFIER.value)
CLASSIFICATION_PERCEPTRON = '{}.{}'.format(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.PERCEPTRON.value)
CLASSIFICATION_XGBOOST = '{}.{}'.format(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.XGBOOST.value)
CLASSIFICATION_NN = '{}.{}'.format(PredictiveModels.CLASSIFICATION.value, ClassificationMethods.NN.value)

UPDATE_INCREMENTAL_NAIVE_BAYES = '{}.{}'.format(JobTypes.UPDATE.value, ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value)
UPDATE_INCREMENTAL_ADAPTIVE_TREE = '{}.{}'.format(JobTypes.UPDATE.value, ClassificationMethods.ADAPTIVE_TREE.value)
UPDATE_INCREMENTAL_HOEFFDING_TREE = '{}.{}'.format(JobTypes.UPDATE.value, ClassificationMethods.HOEFFDING_TREE.value)


class Classification(PredictiveModel):
    """Container of Classification to be shown in frontend"""

    @staticmethod
    def init(configuration: dict) -> PredictiveModel:
        classifier_type = configuration['prediction_method']
        if classifier_type == ClassificationMethods.DECISION_TREE.value:
            default_configuration = classification_decision_tree()
            return DecisionTree.objects.get_or_create(
                prediction_method=classifier_type,
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                max_depth=configuration.get('max_depth', default_configuration['max_depth']),
                min_samples_split=configuration.get('min_samples_split', default_configuration['min_samples_split']),
                min_samples_leaf=configuration.get('min_samples_leaf', default_configuration['min_samples_leaf'])
            )[0]
        elif classifier_type == ClassificationMethods.KNN.value:
            default_configuration = classification_knn()
            return Knn.objects.get_or_create(
                prediction_method=classifier_type,
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                n_neighbors=configuration.get('n_neighbors', default_configuration['n_neighbors']),
                weights=configuration.get('weights', default_configuration['weights'])
            )[0]
        elif classifier_type == ClassificationMethods.RANDOM_FOREST.value:
            default_configuration = classification_random_forest()
            return RandomForest.objects.get_or_create(
                prediction_method=classifier_type,
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                n_estimators=configuration.get('n_estimators', default_configuration['n_estimators']),
                max_depth=configuration.get('max_depth', default_configuration['max_depth']),
                max_features=configuration.get('max_features', default_configuration['max_features'])
            )[0]
        elif classifier_type == ClassificationMethods.XGBOOST.value:
            default_configuration = classification_xgboost()
            return XGBoost.objects.get_or_create(
                prediction_method=classifier_type,
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                n_estimators=configuration.get('n_estimators', default_configuration['n_estimators']),
                max_depth=configuration.get('max_depth', default_configuration['max_depth'])
            )[0]
        elif classifier_type == ClassificationMethods.MULTINOMIAL_NAIVE_BAYES.value:
            default_configuration = classification_incremental_naive_bayes()
            return NaiveBayes.objects.get_or_create(
                prediction_method=classifier_type,
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                alpha=configuration.get('alpha', default_configuration['alpha']),
                fit_prior=configuration.get('fit_prior', default_configuration['fit_prior'])
            )[0]
        elif classifier_type == ClassificationMethods.HOEFFDING_TREE.value:
            default_configuration = classification_incremental_hoeffding_tree()
            return HoeffdingTree.objects.get_or_create(
                prediction_method=classifier_type,
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                grace_period=configuration.get('grace_period', default_configuration['grace_period']),
                split_criterion=configuration.get('split_criterion', default_configuration['split_criterion']),
                split_confidence=configuration.get('split_confidence', default_configuration['split_confidence']),
                tie_threshold=configuration.get('tie_threshold', default_configuration['tie_threshold']),
                remove_poor_atts=configuration.get('remove_poor_atts', default_configuration['remove_poor_atts']),
                leaf_prediction=configuration.get('leaf_prediction', default_configuration['leaf_prediction']),
                nb_threshold=configuration.get('nb_threshold', default_configuration['nb_threshold'])
            )[0]
        elif classifier_type == ClassificationMethods.ADAPTIVE_TREE.value:
            default_configuration = classification_incremental_adaptive_tree()
            return AdaptiveHoeffdingTree.objects.get_or_create(
                prediction_method=classifier_type,
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                grace_period=configuration.get('grace_period', default_configuration['grace_period']),
                split_criterion=configuration.get('split_criterion', default_configuration['split_criterion']),
                split_confidence=configuration.get('split_confidence', default_configuration['split_confidence']),
                tie_threshold=configuration.get('tie_threshold', default_configuration['tie_threshold']),
                remove_poor_atts=configuration.get('remove_poor_atts', default_configuration['remove_poor_atts']),
                leaf_prediction=configuration.get('leaf_prediction', default_configuration['leaf_prediction']),
                nb_threshold=configuration.get('nb_threshold', default_configuration['nb_threshold'])
            )[0]
        elif classifier_type == ClassificationMethods.SGDCLASSIFIER.value:
            default_configuration = classification_incremental_sgd_classifier()
            return SGDClassifier.objects.get_or_create(
                prediction_method=classifier_type,
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                loss=configuration.get('loss', default_configuration['loss']),
                penalty=configuration.get('penalty', default_configuration['penalty']),
                alpha=configuration.get('alpha', default_configuration['alpha']),
                l1_ratio=configuration.get('l1_ratio', default_configuration['l1_ratio']),
                fit_intercept=configuration.get('fit_intercept', default_configuration['fit_intercept']),
                tol=configuration.get('tol', default_configuration['tol']),
                epsilon=configuration.get('epsilon', default_configuration['epsilon']),
                learning_rate=configuration.get('learning_rate', default_configuration['learning_rate']),
                eta0=1 if configuration.get('eta0', default_configuration['eta0']) <= 0 and configuration.get('learning_rate', default_configuration['learning_rate']) in ("constant", "invscaling", "adaptive") else configuration.get('eta0', default_configuration['eta0']),
                power_t=configuration.get('power_t', default_configuration['power_t']),
                n_iter_no_change=configuration.get('n_iter_no_change', default_configuration['n_iter_no_change']),
                validation_fraction=configuration.get('validation_fraction',
                                                      default_configuration['validation_fraction']),
                average=configuration.get('average', default_configuration['average'])
            )[0]
        elif classifier_type == ClassificationMethods.PERCEPTRON.value:
            default_configuration = classification_incremental_perceptron()
            return Perceptron.objects.get_or_create(
                prediction_method=classifier_type,
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                penalty=configuration.get('penalty', default_configuration['penalty']),
                alpha=configuration.get('alpha', default_configuration['alpha']),
                fit_intercept=configuration.get('fit_intercept', default_configuration['fit_intercept']),
                tol=configuration.get('tol', default_configuration['tol']),
                shuffle=configuration.get('shuffle', default_configuration['shuffle']),
                eta0=configuration.get('eta0', default_configuration['eta0']),
                validation_fraction=configuration.get('validation_fraction',
                                                      default_configuration['validation_fraction']),
                n_iter_no_change=configuration.get('n_iter_no_change', default_configuration['n_iter_no_change'])
            )[0]
        elif classifier_type == ClassificationMethods.NN.value:
            default_configuration = classification_nn()
            return NeuralNetwork.objects.get_or_create(
                prediction_method=classifier_type,
                predictive_model=PredictiveModels.CLASSIFICATION.value,
                n_hidden_layers=configuration.get('n_hidden_layers', default_configuration['n_hidden_layers']),
                n_hidden_units=configuration.get('n_hidden_units', default_configuration['n_hidden_units']),
                activation=configuration.get('activation', default_configuration['activation']),
                n_epochs=configuration.get('n_epochs', default_configuration['n_epochs']),
                dropout_rate=configuration.get('dropout_rate', default_configuration['dropout_rate']),
            )[0]
        else:
            raise ValueError('classifier type {} not recognized'.format(classifier_type))


class DecisionTree(Classification):
    max_depth = models.PositiveIntegerField(null=True)
    min_samples_split = models.PositiveIntegerField()
    min_samples_leaf = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf
        }


KNN_WEIGHTS = (
    ('uniform', 'uniform'),
    ('distance', 'distance')
)


class Knn(Classification):
    n_neighbors = models.PositiveIntegerField()
    weights = models.CharField(choices=KNN_WEIGHTS, default='uniform', max_length=max(len(el[1]) for el in KNN_WEIGHTS)+1)

    def to_dict(self):
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights
        }


class RandomForest(Classification):
    n_estimators = models.PositiveIntegerField()
    max_depth = models.PositiveIntegerField(null=True)
    max_features = models.CharField(null=True, default='auto', max_length=10)

    def to_dict(self):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'max_features': self.max_features
        }


class XGBoost(Classification):
    n_estimators = models.PositiveIntegerField()
    max_depth = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth
        }


class NaiveBayes(Classification):
    alpha = models.FloatField()
    fit_prior = models.BooleanField()

    def to_dict(self):
        return {
            'alpha': self.alpha,
            'fit_prior': self.fit_prior
        }


HOEFFDING_TREE_SPLIT_CRITERION = (
    ('gini', 'gini'),
    ('info_gain', 'info_gain')
)

HOEFFDING_TREE_LEAF_PREDICTION = (
    ('mc', 'mc'),
    ('nb', 'nb'),
    ('nba', 'nba')
)


class HoeffdingTree(Classification):
    grace_period = models.PositiveIntegerField()
    split_criterion = models.CharField(choices=HOEFFDING_TREE_SPLIT_CRITERION, default='uniform', max_length=max(len(el[1]) for el in HOEFFDING_TREE_SPLIT_CRITERION)+1)
    split_confidence = models.FloatField()
    tie_threshold = models.FloatField()
    remove_poor_atts = models.BooleanField()
    leaf_prediction = models.CharField(choices=HOEFFDING_TREE_LEAF_PREDICTION, default='uniform', max_length=max(len(el[1]) for el in HOEFFDING_TREE_LEAF_PREDICTION)+1)
    nb_threshold = models.FloatField()

    def to_dict(self):
        return {
            'grace_period': self.grace_period,
            'split_criterion': self.split_criterion,
            'split_confidence': self.split_confidence,
            'tie_threshold': self.tie_threshold,
            'remove_poor_atts': self.remove_poor_atts,
            'leaf_prediction': self.leaf_prediction,
            'nb_threshold': self.nb_threshold
        }


class AdaptiveHoeffdingTree(Classification):
    grace_period = models.PositiveIntegerField()
    split_criterion = models.CharField(choices=HOEFFDING_TREE_SPLIT_CRITERION, default='uniform', max_length=max(len(el[1]) for el in HOEFFDING_TREE_SPLIT_CRITERION)+1)
    split_confidence = models.FloatField()
    tie_threshold = models.FloatField()
    remove_poor_atts = models.BooleanField()
    leaf_prediction = models.CharField(choices=HOEFFDING_TREE_LEAF_PREDICTION, default='uniform', max_length=max(len(el[1]) for el in HOEFFDING_TREE_LEAF_PREDICTION)+1)
    nb_threshold = models.FloatField()

    def to_dict(self):
        return {
            'grace_period': self.grace_period,
            'split_criterion': self.split_criterion,
            'split_confidence': self.split_confidence,
            'tie_threshold': self.tie_threshold,
            'remove_poor_atts': self.remove_poor_atts,
            'leaf_prediction': self.leaf_prediction,
            'nb_threshold': self.nb_threshold
        }


SGDCLASSIFIER_LOSS = (
    ('hinge', 'hinge'),
    ('log', 'log'),
    ('modified_huber', 'modified_huber'),
    ('squared_hinge', 'squared_hinge'),
    ('perceptron', 'perceptron'),
    ('squared_loss', 'squared_loss'),
    ('huber', 'huber'),
    ('epsilon_insensitive', 'epsilon_insensitive'),
    ('squared_epsilon_insensitive', 'squared_epsilon_insensitive')
)

SGDCLASSIFIER_PENALTY = (
    ('l1', 'l1'),
    ('l2', 'l2'),
    ('elasticnet', 'elasticnet')
)

SGDCLASSIFIER_LEARNING_RATE = (
    ('constant', 'constant'),
    ('optimal', 'optimal'),
    ('invscaling', 'invscaling'),
    ('adaptive', 'adaptive')
)


class SGDClassifier(Classification):
    loss = models.CharField(choices=SGDCLASSIFIER_LOSS, default='uniform', max_length=max(len(el[1]) for el in SGDCLASSIFIER_LOSS)+1)
    penalty = models.CharField(choices=SGDCLASSIFIER_PENALTY, default='l1', max_length=max(len(el[1]) for el in SGDCLASSIFIER_PENALTY)+1, null=True)
    alpha = models.FloatField()
    l1_ratio = models.FloatField()
    fit_intercept = models.BooleanField()
    tol = models.FloatField()
    epsilon = models.FloatField()
    learning_rate = models.CharField(choices=SGDCLASSIFIER_LEARNING_RATE, default='constant', max_length=max(len(el[1]) for el in SGDCLASSIFIER_LEARNING_RATE)+1)
    eta0 = models.PositiveIntegerField()
    power_t = models.FloatField()
    n_iter_no_change = models.PositiveIntegerField()
    validation_fraction = models.FloatField()
    average = models.BooleanField()

    def to_dict(self):
        return {
            'loss': self.loss,
            'penalty': self.penalty,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'fit_intercept': self.fit_intercept,
            'tol': self.tol,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'eta0': self.eta0,
            'power_t': self.power_t,
            'n_iter_no_change': self.n_iter_no_change,
            'validation_fraction': self.validation_fraction,
            'average': self.average
        }


PERCEPTRON_PENALTY = (
    ('l1', 'l1'),
    ('l2', 'l2'),
    ('elasticnet', 'elasticnet')
)


class Perceptron(Classification):
    penalty = models.CharField(choices=PERCEPTRON_PENALTY, default='l1', max_length=max(len(el[1]) for el in PERCEPTRON_PENALTY)+1, null=True)
    alpha = models.FloatField()
    fit_intercept = models.BooleanField()
    tol = models.FloatField()
    shuffle = models.BooleanField()
    eta0 = models.PositiveIntegerField()
    validation_fraction = models.FloatField()
    n_iter_no_change = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'penalty': self.penalty,
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'tol': self.tol,
            'shuffle': self.shuffle,
            'eta0': self.eta0,
            'validation_fraction': self.validation_fraction,
            'n_iter_no_change': self.n_iter_no_change
        }


NEURAL_NETWORKS_ACTIVATION = (
    ('sigmoid', 'sigmoid'),
    ('tanh', 'tanh'),
    ('relu', 'relu')
)


class NeuralNetwork(Classification):
    n_hidden_layers = models.PositiveIntegerField()
    n_hidden_units = models.PositiveIntegerField()
    activation = models.CharField(choices=NEURAL_NETWORKS_ACTIVATION, default='relu', max_length=max(len(el[1]) for el in NEURAL_NETWORKS_ACTIVATION)+1)
    n_epochs = models.PositiveIntegerField()
    dropout_rate = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'n_hidden_layers': self.n_hidden_layers,
            'n_hidden_units': self.n_hidden_units,
            'activation': self.activation,
            'n_epochs': self.n_epochs,
            'dropout_rate': self.dropout_rate
        }
