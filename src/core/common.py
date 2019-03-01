"""
common methods used in the core package
"""
from src.jobs.models import Job


def get_method_config(job: Job) -> (str, dict):
    """returns the method configuration dictionary

    :param job: job configuration
    :return: method string and method configuration dict

    """
    method = job.method
    method_conf_name = "{}.{}".format(job.type, method)
    config = job[method_conf_name]
    return method, config
