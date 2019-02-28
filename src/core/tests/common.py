"""
common methods and functionalities for the core tests
"""

import os
import sys

from src.core.default_configuration import CONF_MAP, clustering_kmeans
from src.utils.tests_utils import general_example_filepath, repair_example_filepath, general_example_test_filepath, \
    general_example_train_filepath


class HidePrints:
    """
    hides prints during tests for easier output reading
    """

    def __init__(self):
        self._original_stdout = sys.stdout

    def __enter__(self):
        """
        hides the following prints by redirecting sys.stdout
        """
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """restores the original print behavior

        :param exc_type: TODO: complete
        :param exc_val: TODO: complete
        :param exc_tb: TODO: complete
        """
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
    split['training_log_path'] = general_example_train_filepath
    split['test_log_path'] = general_example_test_filepath
    return split


def repair_example():
    split = dict()
    split['id'] = 1
    split['config'] = dict()
    split['type'] = 'single'
    split['original_log_path'] = repair_example_filepath
    return split


def add_default_config(job: dict, prediction_method=""):
    """Map to job method default config"""
    if prediction_method == "":
        prediction_method = job['type']
    method_conf_name = "{}.{}".format(prediction_method, job['method'])
    method_conf = CONF_MAP[method_conf_name]()
    job[method_conf_name] = method_conf
    job['kmeans'] = clustering_kmeans()
    return job
