from django.db import models

from src.predictive_model.models import PredictiveModel
from src.core.constants import KNN, DECISION_TREE, RANDOM_FOREST, XGBOOST, HOEFFDING_TREE, ADAPTIVE_TREE, SGDCLASSIFIER, \
    PERCEPTRON


class Classification(PredictiveModel):
    """Container of Classification to be shown in frontend"""

    @staticmethod
    def init(classification, configuration=None):
        if classification == DECISION_TREE:
            return DecisionTree.objects.get_or_create(

            )
        elif classification == KNN:
            return Knn.objects.get_or_create(

            )
        elif classification == RANDOM_FOREST:
            return RandomForest.objects.get_or_create(

            )
        elif classification == XGBOOST:
            return XGBoost.objects.get_or_create(

            )
        elif classification == NAIVE_BAYES:
            return NaiveBayes.objects.get_or_create(

            )
        elif classification == HOEFFDING_TREE:
            return HoeffdingTree.objects.get_or_create(

            )
        elif classification == ADAPTIVE_TREE:
            return AdaptiveHoeffdingTree.objects.get_or_create(

            )
        elif classification == SGDCLASSIFIER:
            return SGDClassifier.objects.get_or_create(

            )
        elif classification == PERCEPTRON:
            return Perceptron.objects.get_or_create(

            )
        else:
            raise ValueError('classification ', classification, 'not recognized')

    def to_dict(self):
        return {}


class DecisionTree(Classification):
    max_depth = models.PositiveIntegerField()
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
    weights = models.CharField(choices=KNN_WEIGHTS, default='uniform', max_length=20)

    def to_dict(self):
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights
        }


class RandomForest(Classification):
    n_estimators = models.PositiveIntegerField()
    max_depth = models.PositiveIntegerField()
    max_features = models.PositiveIntegerField()

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
    split_criterion = models.CharField(choices=HOEFFDING_TREE_SPLIT_CRITERION, default='uniform', max_length=20)
    split_confidence = models.FloatField()
    tie_threshold = models.FloatField()
    remove_poor_atts = models.BooleanField()
    leaf_prediction = models.CharField(choices=HOEFFDING_TREE_LEAF_PREDICTION, default='uniform', max_length=20)
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
    split_criterion = models.CharField(choices=HOEFFDING_TREE_SPLIT_CRITERION, default='uniform', max_length=20)
    split_confidence = models.FloatField()
    tie_threshold = models.FloatField()
    remove_poor_atts = models.BooleanField()
    leaf_prediction = models.CharField(choices=HOEFFDING_TREE_LEAF_PREDICTION, default='uniform', max_length=20)
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
    loss = models.CharField(choices=SGDCLASSIFIER_LOSS, default='uniform', max_length=20)
    penalty = models.CharField(choices=SGDCLASSIFIER_PENALTY, default='l1', max_length=20)
    alpha = models.FloatField()
    l1_ratio = models.FloatField()
    fit_intercept = models.BooleanField()
    tol = models.FloatField()
    epsilon = models.FloatField()
    learning_rate = models.CharField(choices=SGDCLASSIFIER_LEARNING_RATE, default='constant', max_length=20)
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
    penalty = models.CharField(choices=PERCEPTRON_PENALTY, default='l1', max_length=20)
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


NEURAL_NETWORKS_ACTIVATION_FUNCTION = (
    ('sigmoid', 'sigmoid'),
    ('tanh', 'tanh'),
    ('relu', 'relu')
)


class NeuralNetworks(Classification):
    hidden_layers = models.PositiveIntegerField()
    hidden_units = models.PositiveIntegerField()
    activation_function = models.CharField(choices=NEURAL_NETWORKS_ACTIVATION_FUNCTION, default='relu', max_length=20)
    epochs = models.PositiveIntegerField()
    dropout_rate = models.PositiveIntegerField()

    def to_dict(self):
        return {
            'hidden_layers': self.hidden_layers,
            'hidden_units': self.hidden_units,
            'activation_function': self.activation_function,
            'epochs': self.epochs,
            'dropout_rate': self.dropout_rate

        }
