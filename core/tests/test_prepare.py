import os
import sys

from core.default_configuration import CONF_MAP, kmeans
from utils.tests_utils import general_example_filepath, repair_example_filepath, general_example_test_filepath, \
    general_example_train_filepath


class HidePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def split_single():
    split = dict()
    split['id'] = 1
    split['config'] = dict()
    split['type'] = 'single'
    split['original_log_path'] = general_example_filepath
    return split


def split_double():
    split = dict()
    split['id'] = 1
    split['config'] = dict()
    split['type'] = 'double'
    split['test_log_path'] = general_example_test_filepath
    split['training_log_path'] = general_example_train_filepath
    return split


def repair_example():
    split = dict()
    split['id'] = 1
    split['config'] = dict()
    split['type'] = 'single'
    split['original_log_path'] = repair_example_filepath
    return split


def add_default_config(job: dict, type=""):
    """Map to job method default config"""
    if type == "":
        type = job['type']
    method_conf_name = "{}.{}".format(type, job['method'])
    method_conf = CONF_MAP[method_conf_name]()
    job[method_conf_name] = method_conf
    job['kmeans'] = kmeans()
    return job
