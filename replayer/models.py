from django.db import models
from jsonfield.fields import JSONField
from core.constants import *
from jobs.models import Job
from training.models import PredModels
import datetime

class Log(models.Model):
    """Container of Log to be shown in frontend"""
    config = models.CharField(default="", null=True, max_length=500)

    def to_dict(self):
        log = dict()
        log['config'] = self.config
        return log

class Trace(models.Model):
    """Container of Log to be shown in frontend"""
    config = models.CharField(default="", null=True, max_length=500)
    log = models.ForeignKey('Log', on_delete=models.DO_NOTHING, related_name='trace_log')
    results=JSONField(default={})
    model =models.ForeignKey(PredModels, on_delete=models.DO_NOTHING, related_name='trace_model', blank=True, null=True, default=None)

    def to_dict(self):
        trace = dict()
        trace['log'] = self.log
        trace['config'] = self.config        
        trace['results'] = self.results
        trace['model'] = self.model
        return trace
    
class Event(models.Model):
    """Container of Log to be shown in frontend"""
    config = models.CharField(default="", null=True, max_length=500)
    trace = models.ForeignKey("Trace", on_delete=models.DO_NOTHING, related_name='trace', blank=True, null=True)
    

    def to_dict(self):
        event = dict()
        event['trace'] = self.trace.id
        event['config'] = self.config
        event['elapsed_time'] = self.elapsed_time
        return event