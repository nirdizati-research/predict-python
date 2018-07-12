import json
import xml.etree.ElementTree as Et
import datetime

from dateutil.parser import parse as dateparser
from opyenxes.factory.XFactory import XFactory
from opyenxes.model.XAttributeMap import XAttributeMap
from opyenxes.out.XesXmlSerializer import XesXmlSerializer

from core.core import runtime_calculate
from encoders.encoding_container import ZERO_PADDING, ALL_IN_ONE
from predModels.models import PredModels
from runtime.models import XTrace, XEvent, XLog, DemoReplayer
from jobs.ws_publisher import publish


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
    tmap = xMap_to_dict(tr.get_attributes())
    tname = str(tmap.get('concept:name'))
    trace_map = json.dumps(tmap)
    xmap = ev.get_attributes()
    event_map = json.dumps(xMap_to_dict(xmap))
    

    log, created = XLog.objects.get_or_create(config=log_map, real_log=real_log)
    try:
        trace = XTrace.objects.get(name = tname, config=trace_map, xlog=log)
        trace.reg_model = reg_model
        trace.class_model = class_model
    except XTrace.DoesNotExist:
        trace = XTrace.objects.create(name = tname, config=trace_map, xlog=log, reg_model=reg_model, class_model=class_model, real_log=real_log.id)

    if end:
        trace.completed = True
        trace.save()
        publish(trace)
        return
    elif trace.completed:
        trace.completed = False
        trace.save()

    try:
        event = XEvent.objects.get(config=event_map, trace=trace)
    except XEvent.DoesNotExist:
        event = XEvent.objects.create(config=event_map, trace=trace, xid=event_xid.__str__())

    events = XEvent.objects.filter(trace=trace, pk__lte=event.id)

    run_log = run.create_log(XAttributeMap(json.loads(log.config)))
    run_trace = run.create_trace(XAttributeMap(json.loads(trace.config)))
    c = 0

    for event in events:
        c = c + 1
        evt = run.create_event(XAttributeMap(json.loads(event.config)))
        run_trace.append(evt)
    run_log.append(run_trace)
    if c == 1:
        trace.first_event = str(xmap.get('time:timestamp'))
    trace.last_event = str(xmap.get('time:timestamp'))
    
    trace.duration = datetime.timedelta.total_seconds(dateparser(str(trace.last_event)) - dateparser(str(trace.first_event)))
    trace.n_events = c
    trace.save()
    
    try:
        if trace.reg_model is not None:
            if trace.reg_model.config['encoding']['padding'] != ZERO_PADDING and trace.reg_model.config['encoding'][
                'generation_type'] != ALL_IN_ONE:
                reg_config = trace.reg_model.config
                reg_config['encoding']['prefix_length'] = c
                right_reg_model = PredModels.objects.filter(config=reg_config)
                trace.reg_model = right_reg_model[0]
            result_data = runtime_calculate(run_log, trace.reg_model.to_dict())
            trace.reg_results = result_data['prediction']
            trace.reg_actual = result_data['label']
            trace.save()
    except Exception as e:
       return print("An exception has occurred in regression, error:" + str(e.__repr__()))
    try:
        if trace.class_model is not None:
            if trace.class_model.config['encoding']['padding'] != ZERO_PADDING and trace.class_model.config['encoding'][
                'generation_type'] != ALL_IN_ONE:
                class_config = trace.class_model.config
                class_config['encoding']['prefix_length'] = c
                right_class_model = PredModels.objects.filter(config=class_config)
                trace.class_model = right_class_model[0]
            result_data = runtime_calculate(run_log, trace.class_model.to_dict())
            trace.class_results = result_data['prediction']
            trace.class_actual = result_data['label']
            trace.save()
    except Exception as e:
        DemoReplayer.objects.filter(pk=replayer_id).update(running=False)
        trace.error = str(e.__repr__())
        trace.save()
        return print("An exception has occurred in classification, error:" + str(e.__repr__()))
    trace.error = ""
    trace.save()
    publish(trace)


def parse(xml):
    element = Et.fromstring(xml.encode("utf-8"))
    return element


def xMap_to_dict(xmap):
    d = dict()
    for key in xmap.keys():
        d[key] = str(xmap.get(key))
    return d
