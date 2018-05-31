from django.db import models
from logs.models import Log
from jsonfield.fields import JSONField
from predModels.models import PredModels

class DemoReplayer(models.Model):
    running = models.BooleanField(default=False)

class XLog(models.Model):
    config = models.CharField(default="", null=True, max_length=500)
    real_log = models.ForeignKey(Log, on_delete=models.DO_NOTHING, related_name='real_log')

    def to_dict(self):
        xlog = dict(self.config)
        xlog['real_log'] = self.real_log
        return xlog
    

class XTrace(models.Model):
    config = models.CharField(default="", null=True, max_length=500)
    xlog = models.ForeignKey('XLog', on_delete=models.DO_NOTHING, related_name='xlog')
    real_log = models.IntegerField()
    reg_results = JSONField(default={})
    class_results = JSONField(default={})
    reg_model =models.ForeignKey(PredModels, on_delete=models.DO_NOTHING, related_name='reg_trace_model', blank=True, null=True, default=None)
    class_model =models.ForeignKey(PredModels, on_delete=models.DO_NOTHING, related_name='class_trace_model', blank=True, null=True, default=None)

    def to_dict(self):
        trace = dict()
        trace['xlog'] = self.xlog
        trace['config'] = self.config
        trace['reg_results'] = self.reg_results
        trace['class_results'] = self.class_results
        trace['reg_model'] = self.reg_model
        trace['class_model'] = self.class_model
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