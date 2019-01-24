from opyenxes.data_in.XUniversalParser import XUniversalParser
from pm4py.objects.log.importer.xes import factory as xes_importer

#TODO substitute all occurrences of this with thew new one
def get_logs_old(file_path: str):
    """Read in event log from disk

    Uses XUniversalParser to parse log.
    :return: The parsed list of logs.
    :rtype: list[XLog]
    """
    print("Reading in log from {}".format(file_path))
    with open(file_path) as file:
        logs = XUniversalParser().parse(file)
    return logs


def get_logs(file_path: str):
    """Read in event log from disk

    Uses xes_importer to parse log.
    """
    print("Reading in log from {}".format(file_path))
    return xes_importer.import_log(file_path)


def save_file(file, path):
    print("Saving uploaded file to {} ".format(path))
    with open(path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
