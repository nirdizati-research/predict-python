from utils.log_metrics import events_by_date, resources_by_date, new_trace_start, trace_attributes, max_events_in_log
from logs.file_service import get_log, save_file
from logs.models import Log


def create_log(file, name: str, folder='log_cache/'):
    path = folder + name
    save_file(file, path)
    properties = create_properties(path)
    return Log.objects.create(name=name, path=path, properties=properties)


def create_properties(path: str):
    """Create read-only dict with methods in this class"""
    print("Creating properties for log {}".format(path))
    logs = get_log(path)
    properties = dict()
    properties["events"] = events_by_date(logs)
    properties["resources"] = resources_by_date(logs)
    properties["maxEventsInLog"] = max_events_in_log(logs)
    properties["traceAttributes"] = trace_attributes(logs)
    properties["newTraces"] = new_trace_start(logs)
    print("Properties created")
    return properties
