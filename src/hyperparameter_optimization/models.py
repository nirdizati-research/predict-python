from enum import Enum

from django.db import models

from src.common.models import CommonModel
from src.hyperparameter_optimization.methods_default_config import hyperparameter_optimization_hyperopt


class HyperparameterOptimizationMethods(Enum):
    HYPEROPT = 'hyperopt'


class HyperparameterOptimization(CommonModel):
    @staticmethod
    def init(configuration: dict = {'type': HyperparameterOptimizationMethods.HYPEROPT.value}):
        hyperparameter_optimizer_type = configuration['type']
        if hyperparameter_optimizer_type == HyperparameterOptimizationMethods.HYPEROPT.value:
            default_configuration = hyperparameter_optimization_hyperopt()
            return HyperOpt.objects.get_or_create(
                max_evaluations=configuration.get('max_evaluations', default_configuration['max_evaluations']),
                performance_metric=configuration.get('performance_metric', default_configuration['performance_metric']),
                algorithm_type=configuration.get('algorithm_type', default_configuration['algorithm_type'])
            )
        else:
            raise ValueError('hyperparameter optimizer type ' + hyperparameter_optimizer_type + ' not recognized')


class HyperOptAlgorithms(Enum):
    RANDOM_SEARCH = 'random_search'
    TPE = 'tpe'


class HyperOptLosses(Enum):
    RMSE = 'rmse'
    MAE = 'mae'
    RSCORE = 'rscore'
    ACC = 'acc'
    F1SCORE = 'f1score'
    AUC = 'auc'
    PRECISION = 'precision'
    RECALL = 'recall'
    TRUE_POSITIVE = 'true_positive'
    TRUE_NEGATIVE = 'true_negative'
    FALSE_POSITIVE = 'false_positive'
    FALSE_NEGATIVE = 'false_negative'
    MAPE = 'mape'


HYPEROPT_ALGORITHM_MAPPINGS = (
    (HyperOptAlgorithms.RANDOM_SEARCH.value, 'random_search'),
    (HyperOptAlgorithms.TPE.value, 'tpe')
)


class HyperOpt(HyperparameterOptimization):
    __name__ = HyperparameterOptimizationMethods.HYPEROPT.value
    max_evaluations = models.PositiveIntegerField()
    performance_metric = models.CharField(default='acc', max_length=20)
    algorithm_type = models.CharField(HYPEROPT_ALGORITHM_MAPPINGS, default='random_search', max_length=20)

    def to_dict(self):
        return {
            'max_evaluations': self.max_evaluations,
            'performance_metric': self.performance_metric,
            'algorithm_type': self.algorithm_type
        }
