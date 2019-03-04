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
            )


class HyperOpt(HyperparameterOptimization):
    max_evaluations = models.PositiveIntegerField()
    performance_metric = models.CharField(default='acc', max_length=20)

    def to_dict(self):
        return {
            'max_evaluations': self.max_evaluations,
            'performance_metric': self.performance_metric
        }
