from log_util.log_metrics import events_by_date, resources_by_date, new_trace_start, trace_attributes, max_events_in_log
from logs.file_service import get_logs
from logs.models import Log


def create_log(file, name: str, folder='log_cache/'):
    # TODO change the naming procedure of the files in order to avoid shadowing between uploads
    path = folder + name
    from logs.file_service import save_file
    save_file(file, path)
    properties = create_properties(path)
    log = Log.objects.create(name=name, path=path, properties=properties)
    return log


def create_properties(path: str):
    """Create read-only dict with methods in this class"""
    print("Creating properties for log {}".format(path))
    logs = get_logs(path)
    properties = dict()
    properties["events"] = events_by_date(logs)
    properties["resources"] = resources_by_date(logs)
    properties["maxEventsInLog"] = max_events_in_log(logs)
    properties["traceAttributes"] = trace_attributes(logs)
    properties["newTraces"] = new_trace_start(logs)
    print("Properties created")
    return properties
