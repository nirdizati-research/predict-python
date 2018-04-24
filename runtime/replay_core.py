import json
from runtime.models import XTrace, XEvent, XLog
from opyenxes.factory.XFactory import XFactory
from opyenxes.out.XesXmlSerializer import XesXmlSerializer
from predModels.models import PredModels
from core.core import runtime_calculate
import xml.etree.ElementTree as Et
from xml.dom import minidom
from core.constants import CLASSIFICATION

def prepare(ev, tr, lg):
    run = XFactory()
    serializer=XesXmlSerializer()
    logtmp=Et.Element("log")
    trtmp=Et.Element("trace")
    evtmp=Et.Element("event")
    
    serializer.add_attributes(logtmp, lg.get_attributes().values())
    serializer.add_attributes(trtmp, tr.get_attributes().values())
    serializer.add_attributes(evtmp, ev.get_attributes().values())
    
    log_config = Et.tostring(logtmp)
    trace_config = Et.tostring(trtmp)
    event_config = Et.tostring(evtmp)

    log,created = XLog.objects.get_or_create(config=log_config)
    trace, created = XTrace.objects.get_or_create(config=trace_config, log=log)
        
    event,created = XEvent.objects.get_or_create(config=event_config, trace=trace)    
    
    events = XEvent.objects.filter(trace=trace, pk__lte=event.id)

    run_log = run.create_log(logtmp)
    run_trace = run.create_trace(trtmp)
    run_log.append(run_trace)
    c=0
    for event in events:
        c=c+1
        evt = run.create_event(evtmp)
        run_trace.append(evt)
    if trace.model is not None:
        try:
            modeldb = PredModels.objects.filter(type=CLASSIFICATION, config__contains={'prefix_length':c})[:1]
        except PredModels.DoesNotExist:
            return print("error")
        trace.model=modeldb
    right_model=trace.model
    
    
    trace.results = runtime_calculate(run_log, right_model.to_dict())
    trace.save()
    return 

def parse(xml):
    element=Et.fromstring(xml.encode("utf-8"))
    return element