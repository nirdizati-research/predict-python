from opyenxes.data_in.XUniversalParser import XUniversalParser


def get_logs(file_path: str):
    """Read in event log from disk

    Uses XUniversalParser to parse log.
    :return: The parsed list of logs.
    :rtype: list[XLog]
    """
    print("Reading in log from {}".format(file_path))
    with open(file_path) as file:
        logs = XUniversalParser().parse(file)
    return logs


def save_file(file, path):
    print("Saving uploaded file to {} ".format(path))
    with open(path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
