from opyenxes.data_in.XUniversalParser import XUniversalParser


def get_logs(file_path):
    """Read in event log from disk

    Uses XUniversalParser to parse log.
    :return: The parsed list of logs.
    :rtype: list[XLog]
    """
    with open(file_path) as file:
        logs = XUniversalParser().parse(file)
    return logs
