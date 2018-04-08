from sklearn.model_selection import train_test_split

from logs.file_service import get_logs

SPLIT_SEQUENTIAL = 'split_sequential'
SPLIT_TEMPORAL = 'split_temporal'
SPLIT_RANDOM = 'split_random'
SPLIT_STRICT_TEMPORAL = 'split_strict_temporal'


def prepare_logs(split: dict):
    """Returns training_log and test_log"""
    if split['type'] == 'single':
        log = get_logs(split['original_log_path'])[0]
        training_log, test_log = split_single_log(split, log)
        print("Loaded single log from {}".format(split['original_log_path']))
    else:
        # Have to use sklearn to convert some internal data types
        training_log, _ = train_test_split(get_logs(split['training_log_path'])[0], test_size=0)
        test_log, _ = train_test_split(get_logs(split['test_log_path'])[0], test_size=0)
        print("Loaded double logs from {} and {}.".format(split['training_log_path'], split['test_log_path']))
    return training_log, test_log


def split_single_log(split: dict, log: list):
    test_size = split['config'].get('test_size', 0.2)
    if test_size <= 0 or test_size >= 1:
        print("Using out of bound split test_size {}. Reverting to default 0.2.".format(test_size))
        test_size = 0.2
    split_type = split['config'].get('split_type', SPLIT_SEQUENTIAL)
    if split_type == SPLIT_TEMPORAL:
        return temporal_split(log, test_size)
    elif split_type == SPLIT_SEQUENTIAL:
        return split_log(log, test_size=test_size, shuffle=False)
    elif split_type == SPLIT_RANDOM:
        return split_log(log, test_size=test_size, random_state=None)
    else:
        raise TypeError("Unknown split type", split_type)


def temporal_split(log: list, test_size: float):
    # split into train and test using temporal split
    # define our own sort that sort by the first event in trace

    # grouped = data.groupby(self.case_id_col)
    # start_timestamps = grouped[self.timestamp_col].min().reset_index()
    # start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
    # train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio * len(start_timestamps))]
    # train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True,
    #                                                                  kind='mergesort')
    # test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True,
    #                                                                  kind='mergesort')

    training_log, test_log = train_test_split(log, test_size=test_size)
    return training_log, test_log


def split_log(log: list, test_size=0.2, random_state=4, shuffle=True):
    training_log, test_log = train_test_split(log, test_size=test_size, random_state=random_state, shuffle=shuffle)
    return training_log, test_log
