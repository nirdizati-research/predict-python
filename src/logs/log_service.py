import logging

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from pm4py.objects.log.exporter.xes.factory import export_log
from pm4py.objects.log.importer.xes.factory import import_log
from pm4py.objects.log.log import EventLog
from src.logs.models import Log
from src.utils.file_service import create_unique_name
from src.utils.log_metrics import events_by_date, resources_by_date, max_events_in_log, trace_attributes, \
    new_trace_start, avg_events_in_log, std_var_events_in_log, trace_ids, traces

logger = logging.getLogger(__name__)


def create_log(log, name: str, folder='cache/log_cache/', import_in_cache=True):
    logger.info('\tCreating new file (' + name + ') in memory')
    if import_in_cache:
        name = create_unique_name(name)
    path = folder + name
    if import_in_cache:
        if isinstance(log, EventLog):
            export_log(log, path)
        else:
            default_storage.save(path, ContentFile(log.read()))
            log = import_log(path)
    else:  # TODO: this might be risky
        if not isinstance(log, EventLog):
            log = import_log(path)
    properties = create_properties(log)
    return Log.objects.create(name=name, path=path, properties=properties)


def create_properties(log: EventLog) -> dict:
    """Create read-only dict with methods in this class"""
    return {
        'events': events_by_date(log),
        'resources': resources_by_date(log),
        'maxEventsInLog': max_events_in_log(log),
        'avgEventsInLog': avg_events_in_log(log),
        'stdVarEventsInLog': std_var_events_in_log(log) if len(log) > 1 else -1,
        'traceAttributes': trace_attributes(log),
        'newTraces': new_trace_start(log),
        'trace_IDs': trace_ids(log)
        # 'alpha_miner_result': vis_factory.apply(*alpha_miner.apply(log)) #TODO ADD alpha miner
    }


def get_log_trace_attributes(log: EventLog) -> list:
    return traces(log)
