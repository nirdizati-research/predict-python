import json
import time
import datetime
from collections import defaultdict
from runtime.models import XTrace, XEvent, XLog, DemoReplayer
from opyenxes.factory.XFactory import XFactory
from opyenxes.id.XID import XID
from opyenxes.out.XesXmlSerializer import XesXmlSerializer
from predModels.models import PredModels
from core.core import runtime_calculate
import xml.etree.ElementTree as Et
from xml.dom import minidom
from opyenxes.model.XAttributeMap import XAttributeMap
from core.constants import CLASSIFICATION
from encoders.encoding_container import ZERO_PADDING


def prepare(ev, tr, lg, replayer_id, reg_id, class_id, real_log, end=False):
    if int(reg_id) > 0:
        reg_model = PredModels.objects.get(pk=reg_id)
    else:
        reg_model = None
    if int(class_id) > 0:
        class_model = PredModels.objects.get(pk=class_id)
    else:
        class_model = None
    run = XFactory()
    serializer = XesXmlSerializer()
    logtmp = Et.Element("log")
    trtmp = Et.Element("trace")
    evtmp = Et.Element("event")
    
    serializer.add_attributes(logtmp, lg.get_attributes().values())
    serializer.add_attributes(trtmp, tr.get_attributes().values())
    serializer.add_attributes(evtmp, ev.get_attributes().values())
    
    log_config = Et.tostring(logtmp)
    trace_config = Et.tostring(trtmp)
    event_config = Et.tostring(evtmp)
    event_xid = ev.get_id()
    
    log_map = json.dumps(xMap_to_dict(lg.get_attributes()))
    trace_map = json.dumps(xMap_to_dict(tr.get_attributes()))
    xmap = ev.get_attributes()
    event_map = json.dumps(xMap_to_dict(xmap))

    log, created = XLog.objects.get_or_create(config=log_map, real_log=real_log)
    try:
        trace = XTrace.objects.get(config=trace_map, xlog=log)
    except XTrace.DoesNotExist:
        trace = XTrace.objects.create(config=trace_map, xlog=log, reg_model=reg_model, class_model=class_model, real_log=real_log.id)
    
    try:    
        event = XEvent.objects.get(config=event_map, trace=trace)    
    except XEvent.DoesNotExist:
        event = XEvent.objects.create(config=event_map, trace=trace, xid=event_xid.__str__())
    
    events = XEvent.objects.filter(trace=trace, pk__lte=event.id)
    
    run_log = run.create_log(XAttributeMap(json.loads(log.config)))
    run_trace = run.create_trace(XAttributeMap(json.loads(trace.config)))
    c = 0
    if end:
        trace.completed = True
        trace.save()
        return
    for event in events:
        c = c + 1
        evt = run.create_event(XAttributeMap(json.loads(event.config)))
        run_trace.append(evt)
    run_log.append(run_trace)
    try:
        if c == 1:
            trace.first_event = xmap['time:timestamp']
        trace.last_event = xmap['time:timestamp']
        trace.n_events = c
        if trace.reg_model is not None:
            if trace.reg_model.config['encoding']['padding'] != ZERO_PADDING:
                reg_config = trace.reg_model.config
                reg_config['encoding']['prefix_length'] = c
                right_reg_model = PredModels.objects.get(config=reg_config)
                trace.reg_model = right_reg_model
            trace.reg_results = runtime_calculate(run_log, trace.reg_model.to_dict())
            trace.save()
        if trace.class_model is not None:
            if trace.class_model.config['encoding']['padding'] != ZERO_PADDING:
                class_config = trace.class_model.config
                class_config['encoding']['prefix_length'] = c
                right_class_model = PredModels.objects.get(config=class_config)
                trace.class_model = right_class_model
            trace.class_results = runtime_calculate(run_log, trace.class_model.to_dict())
            trace.save()
    except PredModels.DoesNotExist:
        DemoReplayer.objects.filter(pk=replayer_id).update(running=False)
        return print("Can't find a suitable model for this trace")
    trace.save()
    return 


def parse(xml):
    element = Et.fromstring(xml.encode("utf-8"))
    return element


def xMap_to_dict(xmap):
    d = dict()
    for key in xmap.keys():
        d[key] = str(xmap.get(key))
    return d
