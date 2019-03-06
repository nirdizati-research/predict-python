from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from pm4py.objects.log.exporter.xes.factory import export_log
from pm4py.objects.log.importer.xes.factory import import_log
from pm4py.objects.log.log import EventLog

from src.logs.models import Log
from src.utils.file_service import create_unique_name
from src.utils.log_metrics import events_by_date, resources_by_date, max_events_in_log, trace_attributes, \
    new_trace_start


def create_log(log, name: str, folder='cache/log_cache/'):
    # just a way to avoid two files with same name shadow each other
    name = create_unique_name(name)
    path = folder + name
    if isinstance(log, EventLog):
        export_log(log, path)
    else:
        default_storage.save(path, ContentFile(log.read()))
        log = import_log(path)
    properties = create_properties(log)
    return Log.objects.create(name=name, path=path, properties=properties)


def create_properties(log: EventLog) -> dict:
    """Create read-only dict with methods in this class"""
    return {
        'events': events_by_date(log),
        'resources': resources_by_date(log),
        'maxEventsInLog': max_events_in_log(log),
        'traceAttributes': trace_attributes(log),
        'newTraces': new_trace_start(log)
    }
