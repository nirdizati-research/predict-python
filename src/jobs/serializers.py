from rest_framework import serializers

from src.evaluation.models import Evaluation
from src.hyperparameter_optimization.models import HyperparameterOptimization
from src.predictive_model.models import PredictiveModel
from .models import Job


class JobSerializer(serializers.ModelSerializer):
    config = serializers.SerializerMethodField()

    def get_config(self, job):
        evaluation = Evaluation.objects.filter(pk=job.evaluation.pk).select_subclasses()[
            0] if job.evaluation is not None else None
        hyperparameter_optimizer = \
            HyperparameterOptimization.objects.filter(pk=job.hyperparameter_optimizer.pk).select_subclasses()[
                0] if job.hyperparameter_optimizer is not None else None
        predictive_model = PredictiveModel.objects.filter(pk=job.predictive_model.pk).select_subclasses()[
            0] if job.predictive_model is not None else None
        return {
            'split': job.split.to_dict() if job.split is not None else None,
            'encoding': job.encoding.to_dict() if job.encoding is not None else None,
            'labelling': job.labelling.to_dict() if job.labelling is not None else None,
            'clustering': job.clustering.to_dict() if job.clustering is not None else None,
            'predictive_model': predictive_model.get_full_dict() if job.predictive_model is not None else None,
            'evaluation': evaluation.get_full_dict() if job.evaluation is not None else None,
            'hyperparameter_optimizer': hyperparameter_optimizer.get_full_dict() if job.hyperparameter_optimizer is not None else None,
            'incremental_train': job.incremental_train.to_dict() if job.incremental_train is not None else None
        }

    class Meta:
        model = Job
        fields = (
            'id',
            'created_date',
            'modified_date',
            'error',
            'status',
            'type',
            'results',
            'config')
