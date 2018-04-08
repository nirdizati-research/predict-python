from sklearn.model_selection import train_test_split

from logs.file_service import get_logs


def prepare_logs(split: dict):
    """Returns training_log and test_log"""
    if split['type'] == 'single':
        log = get_logs(split['original_log_path'])[0]
        training_log, test_log = split_log(log)
        print("Loaded single log from {}".format(split['original_log_path']))
    else:
        # Have to use sklearn to convert some internal data types
        training_log, _ = train_test_split(get_logs(split['training_log_path'])[0], test_size=0)
        test_log, _ = train_test_split(get_logs(split['test_log_path'])[0], test_size=0)
        print("Loaded double logs from {} and {}.".format(split['training_log_path'], split['test_log_path']))
    return training_log, test_log


def split_log(log: list, test_size=0.2, random_state=4):
    training_log, test_log = train_test_split(log, test_size=test_size, random_state=random_state)
    return training_log, test_log
