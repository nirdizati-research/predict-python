from rest_framework import serializers

from .models import Job


class JobSerializer(serializers.ModelSerializer):
    config = serializers.SerializerMethodField()

    def get_config(self, job):
        return {
            'split': job.split.to_dict() if job.split is not None else None,
            'encoding': job.encoding.to_dict() if job.encoding is not None else None,
            'labelling': job.labelling.to_dict() if job.labelling is not None else None,
            'clustering': job.clustering.to_dict() if job.clustering is not None else None,
            'predictive_model': job.predictive_model.to_dict() if job.predictive_model is not None else None,
            'evaluation': job.evaluation.to_dict() if job.evaluation is not None else None,
            'hyperparameter_optimizer': job.hyperparameter_optimizer.to_dict() if job.hyperparameter_optimizer is not None else None,
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
            'config')
