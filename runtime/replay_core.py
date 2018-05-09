import json
from runtime.models import XTrace, XEvent, XLog, DemoReplayer
from opyenxes.factory.XFactory import XFactory
from opyenxes.out.XesXmlSerializer import XesXmlSerializer
from predModels.models import PredModels
from core.core import runtime_calculate
import xml.etree.ElementTree as Et
from xml.dom import minidom
from core.constants import CLASSIFICATION

def prepare(ev, tr, lg, replayer_id):
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
    try:
        models = PredModels.objects.filter(type='regression')
        modeldb=models[0]
    except PredModels.DoesNotExist:
        return print("error")
    trace.model=modeldb
    right_model=trace.model
    
    try:
        trace.results = runtime_calculate(run_log, right_model.to_dict())
        trace.save()
    except Exception as e:
        DemoReplayer.objects.filter(pk=replayer_id).update(running=False)
        print("I can't predict this trace because I don't have a suitable model")
        print("Error:" + str(e))
    return 

def parse(xml):
    element=Et.fromstring(xml.encode("utf-8"))
    return element