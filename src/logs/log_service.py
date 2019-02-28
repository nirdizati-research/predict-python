import time

from src.logs.models import Log
from src.utils.file_service import get_log, save_file
from src.utils.log_metrics import events_by_date, resources_by_date, max_events_in_log, trace_attributes, \
    new_trace_start


def create_log(file, name: str, folder='cache/log_cache/'):
    # just a way to avoid two files with same name shadow each other
    name = name.replace('.', '_' + str(time.time()).replace('.', '') + '.')
    path = folder + name
    save_file(file, path)
    properties = create_properties(path)
    return Log.objects.create(name=name, path=path, properties=properties)


def create_properties(path: str) -> dict:
    """Create read-only dict with methods in this class"""
    print("Creating properties for log {}".format(path))
    logs = get_log(path)
    return {
        'events': events_by_date(logs),
        'resources': resources_by_date(logs),
        'maxEventsInLog': max_events_in_log(logs),
        'traceAttributes': trace_attributes(logs),
        'newTraces': new_trace_start(logs)
    }
