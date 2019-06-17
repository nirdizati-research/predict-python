import csv
import time

from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.log.log import EventLog

from src.logs.models import Log

import logging
logger = logging.getLogger(__name__)


def get_log(log: Log) -> EventLog:
    """Read in event log from disk

    Uses xes_importer to parse log.
    """
    filepath = log.path
    logger.info("\t\tReading in log from {}".format(filepath))
    return xes_importer.import_log(filepath)


def create_unique_name(name: str) -> str:
    return name.replace('.', '_' + str(time.time()).replace('.', '') + '.')


# def save_file(file, path):
#     print("Saving uploaded file to {} ".format(path))
#     with open(path, 'wb+') as destination:
#         for chunk in file.chunks():
#             destination.write(chunk)


def save_result(results: dict, job, start_time: float):
    result = [
        results['f1score'],
        results['acc'],
        results['true_positive'],
        results['true_negative'],
        results['false_negative'],
        results['false_positive'],
        results['precision'],
        results['recall'],
        results['auc']
    ]
    result += [job['encoding'][index] for index in range(len(job['encoding']))]
    result += [job['labelling'][index] for index in range(len(job['labelling']))]
    if 'incremental_train' in job:
        result += [job['incremental_train'][index] for index in job['incremental_train'].keys()]
    if 'hyperopt' in job:
        result += [job['hyperopt'][index] for index in job['hyperopt'].keys()]
    result += [job['clustering']]
    result += [job['split'][index] for index in job['split'].keys()]
    result += [job['type']]
    result += [job[job['type'] + '.' + job['method']][index] for index in job[job['type'] + '.' + job['method']].keys()]
    result += [str(time.time() - start_time)]

    with open('results/' + job['type'] + '-' + job['method'] + '_result.csv', 'a+') as log_result_file:
        writer = csv.writer(log_result_file)
        if sum(1 for _ in open('results/' + job['type'] + '-' + job['method'] + '_result.csv')) == 0:
            writer.writerow(['f1score',
                             'acc',
                             'true_positive',
                             'true_negative',
                             'false_negative',
                             'false_positive',
                             'precision',
                             'recall',
                             'auc'] +
                            list(job['encoding']._fields) +
                            list(job['labelling']._fields) +
                            (list(job['incremental_train'].keys()) if 'incremental_train' in job else []) +
                            (list(job['hyperopt'].keys()) if 'hyperopt' in job else []) +
                            ['clustering'] +
                            list(job['split'].keys()) +
                            ['type'] +
                            list(job[job['type'] + '.' + job['method']].keys()) +
                            ['time_elapsed(s)']
                            )
        writer.writerow(result)
