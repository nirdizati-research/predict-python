from django.db import models
from jsonfield.fields import JSONField

from pred_models.models import PredModels
from src.common.models import CommonModel
from src.logs.models import Log


class DemoReplayer(CommonModel):
    running = models.BooleanField(default=False)

    def to_dict(self):
        return {
            'running': self.running
        }


class XLog(CommonModel):
    config = models.CharField(default="", null=True, max_length=500)
    real_log = models.ForeignKey(Log, on_delete=models.DO_NOTHING, related_name='real_log')

    def to_dict(self):
        xlog = dict(self.config)
        xlog['real_log'] = self.real_log
        return xlog


class XTrace(CommonModel):
    config = models.CharField(default="", null=True, max_length=500)
    name = models.CharField(default="", null=True, max_length=30)
    xlog = models.ForeignKey('XLog', on_delete=models.DO_NOTHING, related_name='xlog')
    completed = models.BooleanField(default=False)
    first_event = models.DateTimeField(null=True)
    last_event = models.DateTimeField(null=True)
    n_events = models.IntegerField(default=0)
    error = models.CharField(default="", null=True, max_length=500)
    real_log = models.IntegerField()
    reg_results = JSONField(default={})
    class_results = JSONField(default={})
    reg_actual = JSONField(default={})
    duration = models.IntegerField(default=0)
    class_actual = JSONField(default={})
    reg_model = models.ForeignKey(PredModels, on_delete=models.DO_NOTHING, related_name='reg_trace_model', blank=True,
                                  null=True, default=None)
    class_model = models.ForeignKey(PredModels, on_delete=models.DO_NOTHING, related_name='class_trace_model',
                                    blank=True, null=True, default=None)

    def to_dict(self):
        trace = dict()
        trace['xlog'] = self.xlog
        trace['name'] = self.name
        trace['config'] = self.config
        trace['completed'] = self.completed
        trace['first_event'] = self.first_event
        trace['last_event'] = self.last_event
        trace['n_events'] = self.n_events
        trace['reg_results'] = self.reg_results
        trace['error'] = self.reg_results
        trace['class_results'] = self.class_results
        trace['reg_actual'] = self.reg_results
        trace['class_actual'] = self.class_results
        trace['reg_model'] = self.reg_model
        trace['class_model'] = self.class_model
        return trace


class XEvent(CommonModel):
    config = models.CharField(default="", null=True, max_length=500)
    xid = models.CharField(default="", max_length=200)
    trace = models.ForeignKey("XTrace", on_delete=models.DO_NOTHING, related_name='xtrace', blank=True, null=True)

    def to_dict(self):
        event = dict()
        event['trace'] = self.trace.id
        event['xid'] = self.xid
        event['config'] = self.config
        return event
