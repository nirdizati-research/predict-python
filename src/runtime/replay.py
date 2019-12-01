import json
import logging
import requests

from pm4py.objects.log.exporter.xes.factory import export_log_as_string
from pm4py.objects.log.log import EventLog, Trace
from pm4py.algo.filtering.log.timestamp import timestamp_filter

from src.encoding.common import encode_label_logs
from src.jobs.models import Job, JobTypes
from src.split.splitting import get_train_test_log
from src.utils.django_orm import duplicate_orm_row
from src.utils.file_service import get_log

logger = logging.getLogger(__name__)

def replay_core(replay_job: Job, training_initial_job: Job) -> list:
    """The function create a set with timestamps of events, then create a list of requests
        simulating the log in the time passing

        :param replay_job: job dictionary
        :param training_initial_job: job dictionary
        :return: List of requests
    """

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

    for t in times[2::5]:
        filtered_eventlog = timestamp_filter.apply_events(eventlog, times[0].replace(tzinfo=None),
                                                          t.replace(tzinfo=None))
        trace_list = list()
        event_number = dict()
        for trace in filtered_eventlog:
            trace_list.append(trace.attributes['concept:name'])
            event_number[trace.attributes['concept:name']] = len(trace)
        replay_job.case_id = trace_list
        replay_job.event_number = event_number
        replay_job.save()
        try: #TODO check logger usage
            logger.info("Sending request for replay_prediction task.")
            r = requests.post(
                url="http://server:8000/runtime/replay_prediction/",
                data=export_log_as_string(filtered_eventlog),
                params={'jobId': replay_job.id, 'training_job': training_initial_job.id},
                headers={'Content-Type': 'text/plain', 'charset': 'UTF-8'}
            )
            requests_list.append(str(r))
        except Exception as e:
            requests_list.append(str(e))
            logger.warning(str(e))

    training_log, test_log, additional_columns = get_train_test_log(replay_job.split)
    training_df, _ = encode_label_logs(training_log, test_log, replay_job, additional_columns=additional_columns)

    gold_values = dict(zip(training_df['trace_id'], training_df['label']))
    parent_id = replay_job.id
    final_job = duplicate_orm_row(replay_job)
    final_job.parent_job = Job.objects.filter(pk=parent_id)[0]
    final_job.gold_value = gold_values
    final_job.type = JobTypes.REPLAY_PREDICT.value
    final_job.save()
    return requests_list
