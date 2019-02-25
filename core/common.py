"""
common methods used in the core package
"""


def get_method_config(job: dict):  # -> TODO: complete
    """returns the method configuration dictionary

    :param job: job configuration
    :return: method string and method configuration

    """
    method = job['method']
    method_conf_name = "{}.{}".format(job['type'], method)
    config = job[method_conf_name]
    return method, config
