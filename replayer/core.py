import json
from .models import Trace, Event, Log
from logs.models import Split, Log as Log_Log
from jobs.models import Job
from training.tr_core import calculate as tr_calculate
from opyenxes.factory.XFactory import XFactory
from opyenxes.out.XesXmlSerializer import XesXmlSerializer
from encoders.log_util import elapsed_time
from training.models import PredModels
from encoders.common import encode_run_logs, encode_one_training_logs
from core.regression import regression_run
from core.classification import classifier_run
from core.next_activity import next_activity_run
from replayer.models import Trace, Event
from core.constants import CLASSIFICATION, REGRESSION, NEXT_ACTIVITY
from jobs.tasks import calculate
import xml.etree.ElementTree as Et
from xml.dom import minidom

def prepare(ev, tr, lg):
    #log=Log.objects.get(name=tr['log_name'], path=tr['log_path'])    
    #el_time = elapsed_time(event['trace'], event['event'])
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

    try:
        log = Log.objects.get(config=log_config)
    except Log.DoesNotExist:
        log = Log.objects.create(config=log_config)
    try:
        trace = Trace.objects.get(config=trace_config, log=log)
    except Trace.DoesNotExist:
        trace = Trace.objects.create(config=trace_config, log=log)
        
    try:
        event = Event.objects.get(config=event_config, trace=trace)
    except Event.DoesNotExist:  
        event = Event.objects.create(config=event_config, trace=trace)
    
    
    events = Event.objects.filter(trace=trace)

    run_log = run.create_log(parse(log.config))
    run_trace = run.create_trace(parse(trace.config))
    run_log.append(run_trace)
    for event in events:
        evt = run.create_event(parse(event.config))
        run_trace.append(evt)
    if trace.model==None:
        model_set = PredModels.objects.filter(prefix_length=len(events))
        if len(model_set) == 0:
            return print("error")
        else:
            modeldb=model_set[0]
            trace.model=modeldb
    right_model=trace.model
    trace.results = runtime(run_log, right_model.to_dict())
    trace.save()
    return 

def parse(xml):
    element=Et.fromstring(xml)
    return element
    

def runtime(run_log, model):
    """ Main entry method for calculations"""
    #print("Start job {} with {}".format(job['type'], get_run(job)))
    tr_log = Log_Log.objects.get(name=model['log_name'],path=model['log_path'])
    # Python dicts are bad
    run_df, prefix_length= encode_run_logs(run_log, model['encoding'], model['type'])

    try:
        right_model=PredModels.objects.get(encoding=model['encoding'],type=model['type'], method=model['method'],
                                           log=tr_log, prefix_length=prefix_length)
    except PredModels.DoesNotExist:
        split = model['split']
        if split['type'] == 'single':
            clust='noCluster'
        else:
            clust='Kmeans'
        config = {'key': 123,
                       'method': model['method'],
                       'encoding': model['encoding'],
                       'clustering': clust,
                       'prefix_length':prefix_length,
                       "rule": "remaining_time",
                       'threshold': 'default',
                       }
        try:
            split = Split.objects.get(type = 'single', original_log = tr_log)
        except Split.DoesNotExist:
            split = Split.objects.create(type = 'single', original_log = tr_log)
        j=Job.objects.create(config=config, split=split, type=model['type'])
        right_model = tr_calculate(j.to_dict(), redo=True)

    if model['type'] == CLASSIFICATION:
        results = classifier_run(run_df, right_model.to_dict())
    elif model['type'] == REGRESSION:
        results= regression_run(run_df, right_model.to_dict())
    elif model['type'] == NEXT_ACTIVITY:
        results = next_activity_run(run_df, right_model.to_dict())
    else:
        raise ValueError("Type not supported", job['type'])
    #print("End job {}, {} . Results {}".format(job['type'], get_run(job), results))
    return results