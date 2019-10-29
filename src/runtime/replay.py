import sys

import requests
from pm4py.objects.log.exporter.xes.factory import export_log_as_string
from pm4py.objects.log.log import EventLog, Trace
from pm4py.algo.filtering.log.timestamp import timestamp_filter

from src.jobs.models import Job
from src.utils.file_service import get_log


def replay_core(replay_job: Job, training_initial_job: Job) -> list:

    split = replay_job.split
    log = get_log(split.train_log)
    requests_list = list()

    eventlog = EventLog()
    for key in log.attributes.keys():
        eventlog.attributes[key] = log.attributes[key]
    for trace in log:
        new_trace = Trace(trace)
        for key in trace.attributes:
            new_trace.attributes[key] = trace.attributes[key]
        eventlog.append(new_trace)

    times = sorted(set([event['time:timestamp'] for trace in eventlog for event in trace]))

    for t in times[2:]:
        filtered_eventlog = timestamp_filter.apply_events(eventlog, times[0].replace(tzinfo=None),
                                                          t.replace(tzinfo=None))
        try:
            r = requests.post("http://localhost:80/runtime/replay_prediction/",  # TODO: using static address docker mapping
                              data={'log': export_log_as_string(filtered_eventlog),
                                    'modelId': replay_job.id,
                                    'training_job': training_initial_job.id})
            requests_list.append(str(r))
        except requests.ConnectionError as e:
            requests_list.append(str(e))

    return requests_list
