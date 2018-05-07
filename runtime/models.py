from django.db import models
from jsonfield.fields import JSONField
from predModels.models import PredModels

class DemoReplayer(models.Model):
    running = models.BooleanField(default=False)

class XLog(models.Model):
    config = models.CharField(default="", null=True, max_length=500)

    def to_dict(self):
        log = dict(self.config)
        return log
    

class XTrace(models.Model):
    config = models.CharField(default="", null=True, max_length=500)
    log = models.ForeignKey('XLog', on_delete=models.DO_NOTHING, related_name='xlog')
    results=JSONField(default={})
    model =models.ForeignKey(PredModels, on_delete=models.DO_NOTHING, related_name='trace_model', blank=True, null=True, default=None)

    def to_dict(self):
        trace = dict()
        trace['log'] = self.log
        trace['config'] = self.config        
        trace['results'] = self.results
        trace['model'] = self.model
        return trace
    
    
class XEvent(models.Model):
    config = models.CharField(default="", null=True, max_length=500)
    trace = models.ForeignKey("XTrace", on_delete=models.DO_NOTHING, related_name='xtrace', blank=True, null=True)
    

    def to_dict(self):
        event = dict()
        event['trace'] = self.trace.id
        event['config'] = self.config
        event['elapsed_time'] = self.elapsed_time
        return event