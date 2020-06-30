import logging
import pathlib

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from pm4py.objects.log.exporter.xes.factory import export_log as export_log_xes
from pm4py.objects.log.exporter.csv.factory import export_log as export_log_csv
from pm4py.objects.log.importer.xes.factory import import_log as import_log_xes
from pm4py.objects.log.importer.csv.factory import import_event_stream
from pm4py.objects.conversion.log import factory as conversion_factory
from pm4py.util import constants
from pm4py.objects.log.log import EventLog

from src.logs.models import Log
from src.utils.file_service import create_unique_name
from src.utils.log_metrics import events_by_date, resources_by_date, max_events_in_log, trace_attributes, \
    new_trace_start, avg_events_in_log, std_var_events_in_log, trace_ids_in_log, traces_in_log

logger = logging.getLogger(__name__)


def import_log_csv(path):
    return conversion_factory.apply(
        import_event_stream(path),                           # https://pm4py.fit.fraunhofer.de/documentation/1.2
        parameters={constants.PARAMETER_CONSTANT_CASEID_KEY: "case:concept:name",     # this tells the importer
                    constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "concept:name",        # how to parse the csv
                    constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp"}     # and which are the caseID
    )                                                                                 # concept name and timestamp

import_log = {
    '.csv': import_log_csv,
    '.xes': import_log_xes
}

export_log = {
    '.csv': export_log_csv,
    '.xes': export_log_xes
}


def create_log(log, name: str, folder='cache/log_cache/', import_in_cache=True):
    logger.info('\tCreating new file (' + name + ') in memory')
    if import_in_cache:
        name = create_unique_name(name)
    path = folder + name
    if import_in_cache:
        if isinstance(log, EventLog):
            export_log[pathlib.Path(name).suffixes[0]](log, path)
        else:
            default_storage.save(path, ContentFile(log.read()))
            log = import_log[pathlib.Path(name).suffixes[0]](path)
    else:  # TODO: this might be risky
        if not isinstance(log, EventLog):
            log = import_log[pathlib.Path(name).suffixes[0]](path)
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
        'trace_IDs': trace_ids_in_log(log)
        # 'alpha_miner_result': vis_factory.apply(*alpha_miner.apply(log)) #TODO ADD alpha miner
    }


def get_log_trace_attributes(log: EventLog) -> list:
    return traces_in_log(log)


def get_log(log: Log) -> EventLog:
    """Read in event log from disk

    Uses xes_importer to parse log.
    """
    filepath = log.path
    logger.info("\t\tReading in log from {}".format(filepath))
    return import_log[pathlib.Path(log.name).suffixes[0]](filepath)
