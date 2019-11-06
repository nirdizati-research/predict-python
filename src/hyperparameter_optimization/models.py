from enum import Enum

from django.db import models
from model_utils.managers import InheritanceManager

from src.common.models import CommonModel
from src.hyperparameter_optimization.methods_default_config import hyperparameter_optimization_hyperopt


class HyperparameterOptimizationMethods(Enum):
    HYPEROPT = 'hyperopt'
    NONE = 'none'


HYPERPARAMETER_OPTIMIZATION_METHOD = (
    (HyperparameterOptimizationMethods.HYPEROPT.value, 'hyperopt'),
    (HyperparameterOptimizationMethods.NONE.value, 'none'),
)


class HyperparameterOptimization(CommonModel):
    optimization_method = models.CharField(choices=HYPERPARAMETER_OPTIMIZATION_METHOD,
                                           default='hyperopt',
                                           max_length=20)
    elapsed_time = models.DurationField(null=True)
    objects = InheritanceManager()

    @staticmethod
    def init(configuration: dict = {'type': HyperparameterOptimizationMethods.HYPEROPT.value}):
        hyperparameter_optimizer_type = configuration.get('type', HyperparameterOptimizationMethods.NONE.value)
        if hyperparameter_optimizer_type == HyperparameterOptimizationMethods.HYPEROPT.value:
            default_configuration = hyperparameter_optimization_hyperopt()
            return HyperOpt.objects.get_or_create(
                elapsed_time=configuration['elapsed_time'] if 'elapsed_time' in configuration and configuration['elapsed_time'] != '--' else None,
                optimization_method=hyperparameter_optimizer_type,
                max_evaluations=configuration.get('max_evaluations', default_configuration['max_evaluations']),
                performance_metric=configuration.get('performance_metric', default_configuration['performance_metric']),
                algorithm_type=configuration.get('algorithm_type', default_configuration['algorithm_type'])
            )[0]
        elif hyperparameter_optimizer_type == HyperparameterOptimizationMethods.NONE.value:
            return HyperparameterOptimization.objects.get_or_create(
                optimization_method=HyperparameterOptimizationMethods.NONE.value)[0]
        else:
            raise ValueError('hyperparameter optimizer type {} not recognized'.format(hyperparameter_optimizer_type))

    def to_dict(self) -> dict:
        return {
            'elapsed_time': self.elapsed_time
        }


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

HYPEROPT_LOSS_MAPPINGS = (
    (HyperOptLosses.RMSE.value, 'rmse'),
    (HyperOptLosses.MAE.value, 'mae'),
    (HyperOptLosses.RSCORE.value, 'rscore'),
    (HyperOptLosses.ACC.value, 'acc'),
    (HyperOptLosses.F1SCORE.value, 'f1score'),
    (HyperOptLosses.AUC.value, 'auc'),
    (HyperOptLosses.PRECISION.value, 'precision'),
    (HyperOptLosses.RECALL.value, 'recall'),
    (HyperOptLosses.TRUE_POSITIVE.value, 'true_positive'),
    (HyperOptLosses.TRUE_NEGATIVE.value, 'true_negative'),
    (HyperOptLosses.FALSE_POSITIVE.value, 'false_positive'),
    (HyperOptLosses.FALSE_NEGATIVE.value, 'false_negative'),
    (HyperOptLosses.MAPE.value, 'mape')
)


class HyperOpt(HyperparameterOptimization):
    max_evaluations = models.PositiveIntegerField()
    performance_metric = models.CharField(choices=HYPEROPT_LOSS_MAPPINGS, default='acc', max_length=max(len(el[1]) for el in HYPEROPT_LOSS_MAPPINGS)+1)
    algorithm_type = models.CharField(choices=HYPEROPT_ALGORITHM_MAPPINGS, default='tpe', max_length=max(len(el[1]) for el in HYPEROPT_ALGORITHM_MAPPINGS)+1)

    def to_dict(self):
        return {
            'elapsed_time': self.elapsed_time,
            'max_evaluations': self.max_evaluations,
            'performance_metric': self.performance_metric,
            'algorithm_type': self.algorithm_type
        }
