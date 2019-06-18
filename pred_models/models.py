from django.db import models
from jsonfield.fields import JSONField

from src.clustering.models import ClusteringMethods
from src.common.models import CommonModel
from src.jobs.models import JOB_TYPE_MAPPINGS
from src.labelling.label_container import LabelContainer
from src.logs.models import Log

ENC_TYPES = (
    (ClusteringMethods.KMEANS, 'kmeans'),
    (ClusteringMethods.NO_CLUSTER, 'noCluster'),
)


class ModelSplit(CommonModel):
    type = models.CharField(choices=ENC_TYPES, default='noCluster', max_length=20)
    model_path = models.CharField(default='error', max_length=200)
    clusterer_path = models.CharField(blank=True, null=True, max_length=200)
    predtype = models.CharField(choices=JOB_TYPE_MAPPINGS, max_length=max(len(el[1]) for el in JOB_TYPE_MAPPINGS)+1, default='Prediction')

    def to_dict(self):
        split = dict()
        split['type'] = self.type
        split['model_path'] = self.model_path
        if self.type == 'double':
            split['clusterer_path'] = self.clusterer_path
        split['predtype'] = self.predtype
        return split


class PredModels(CommonModel):
    split = models.ForeignKey('ModelSplit', on_delete=models.DO_NOTHING, related_name='split', blank=True, null=True)
    type = models.CharField(choices=JOB_TYPE_MAPPINGS, max_length=max(len(el[1]) for el in JOB_TYPE_MAPPINGS)+1)
    log = models.ForeignKey(Log, on_delete=models.DO_NOTHING, related_name='log', blank=True, null=True)
    config = JSONField(default={})

    def to_dict(self):
        model = dict(self.config)
        model['type'] = self.type
        model['log_path'] = self.log.path
        model['log_name'] = self.log.name
        model['split'] = self.split.to_dict()
        model['label'] = LabelContainer(**self.config['label'])
        # model['encoding'] = EncodingContainer(**self.config['encoding'])
        return model
