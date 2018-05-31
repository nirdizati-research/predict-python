import json
from runtime.models import XTrace, XEvent, XLog, DemoReplayer
from opyenxes.factory.XFactory import XFactory
from opyenxes.out.XesXmlSerializer import XesXmlSerializer
from predModels.models import PredModels
from core.core import runtime_calculate
import xml.etree.ElementTree as Et
from xml.dom import minidom
from core.constants import CLASSIFICATION

ZERO_PADDING = 'zero_padding'

def prepare(ev, tr, lg, replayer_id, reg_id, class_id, real_log):
    if int(reg_id) > 0:
        reg_model=PredModels.objects.get(pk=reg_id)
    else:
        reg_model=None
    if int(class_id) > 0:
        class_model=PredModels.objects.get(pk=class_id)
    else:
        class_model=None
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

    log,created = XLog.objects.get_or_create(config=log_config, real_log = real_log)
    try:
        trace= XTrace.objects.get(config=trace_config, xlog=log)
    except XTrace.DoesNotExist:
        trace= XTrace.objects.create(config=trace_config, xlog=log, reg_model=reg_model, class_model=class_model, real_log = real_log.id)
        
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
        if trace.reg_model is not None:
            reg_config=trace.reg_model.config
            reg_config['encoding']['prefix_length']=c
            if reg_config['encoding']['padding'] != ZERO_PADDING:
                right_reg_model = PredModels.objects.get(config=reg_config)
                trace.reg_model=right_reg_model
            trace.reg_results = runtime_calculate(run_log, trace.reg_model.to_dict())
            trace.save()
        if trace.class_model is not None:
            class_config=trace.class_model.config        
            class_config['encoding']['prefix_length']=c
            if class_config['encoding']['padding'] != ZERO_PADDING:
                right_class_model = PredModels.objects.get(config=class_config)
                trace.class_model=right_class_model
            trace.class_results = runtime_calculate(run_log, trace.class_model.to_dict())
            trace.save()
    except PredModels.DoesNotExist:
        DemoReplayer.objects.filter(pk=replayer_id).update(running=False)
        return print("Can't find a suitable model for this trace")
    trace.save()
    return 

def parse(xml):
    element=Et.fromstring(xml.encode("utf-8"))
    return element