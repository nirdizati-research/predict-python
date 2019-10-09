import requests
from pm4py.objects.log.exporter.xes.factory import export_log_as_string
from pm4py.objects.log.log import EventLog, Trace

from src.jobs.models import Job
from src.utils.file_service import get_log


def replay_core(job: Job) -> None:

    split = job.split
    log = get_log(split.train_log)
    #TODO: for su tutte le tracce
    for trace in log[0:3]:
        eventlog = EventLog()
        for key in log.attributes.keys():
            eventlog.attributes[key] = log.attributes[key]
        trace = Trace(trace)
        for key in log[0].attributes.keys():
            trace.attributes[key] = log[0].attributes[key]
        eventlog.append(trace)

        r = requests.post("http://localhost:8000/runtime/replayprediction/", data={'log': export_log_as_string(eventlog),
                                                                         'modelId': job.id})


