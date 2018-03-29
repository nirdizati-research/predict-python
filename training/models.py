from django.db import models
from jsonfield.fields import JSONField
from core.constants import *
from jobs.models import Job
from logs.models import Log
from logs.file_service import get_logs

TYPES = (
    (CLASSIFICATION, 'Classification'),
    (REGRESSION, 'Regression'),
    (NEXT_ACTIVITY, 'Next activity'),
)

ENCODING = (
    (SIMPLE_INDEX, 'simpleIndex'),
    (BOOLEAN, 'boolean'),
    (FREQUENCY, 'frequency'),
    (COMPLEX, 'complex'),
    (LAST_PAYLOAD, 'lastPayload'),
)

METHODS = (
    (KNN, 'knn'),
    (RANDOM_FOREST, 'randomForest'),
    (DECISION_TREE, 'decisionTree'),
    (LINEAR, 'linear'),
    (LASSO, 'lasso'),   
)



class PredModels(models.Model):
    """Container of Log to be shown in frontend"""
    split = models.ForeignKey('Split', on_delete=models.DO_NOTHING, related_name='log', blank=True, null=True)
    type = models.CharField(choices=TYPES, max_length=20)
    log = models.ForeignKey(Log, on_delete=models.DO_NOTHING, related_name='log', blank=True, null=True)
    prefix_length = models.IntegerField(default=1)
    encoding=models.CharField(choices=ENCODING, max_length=20, default='simpleIndex')
    method=models.CharField(choices=METHODS, max_length=20, default='linear')
    
    
    def to_dict(self):
        model = dict()
        model['type'] = self.type
        model['log_path'] = self.log.path
        model['log_name'] = self.log.name   
        model['prefix_length'] = self.prefix_length
        model['encoding'] = self.encoding
        model['method'] = self.method
        model['split'] = self.split.to_dict()     
        return model

class Split(models.Model):
    """Container of Log to be shown in frontend"""
    type = models.CharField(choices=TYPES, default='single', max_length=20)
    model_path = models.CharField(default='error', max_length=200)
    kmean_path = models.CharField(blank=True, null=True, max_length=200)
    predtype = models.CharField(choices=TYPES, max_length=20, default='Classification')

    def to_dict(self):
        split = dict()
        split['type'] = self.type
        split['model_path'] = self.model_path
        if self.type == 'double':
            split['kmean_path'] = self.kmean_path
        split['predtype'] = self.predtype
        return split