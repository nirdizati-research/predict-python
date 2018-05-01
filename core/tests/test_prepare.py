from jobs.job_creator import CONF_MAP, _kmeans


def split_single():
    split = dict()
    split['id'] = 1
    split['config'] = dict()
    split['type'] = 'single'
    split['original_log_path'] = 'log_cache/general_example.xes'
    return split


def split_double():
    split = dict()
    split['id'] = 1
    split['config'] = dict()
    split['type'] = 'double'
    split['test_log_path'] = 'log_cache/general_example_test.xes'
    split['training_log_path'] = 'log_cache/general_example_training.xes'
    return split


def repair_example():
    split = dict()
    split['id'] = 1
    split['config'] = dict()
    split['type'] = 'single'
    split['original_log_path'] = 'log_cache/repairExample.xes'
    return split


def add_default_config(job: dict, type=""):
    """Map to job method default config"""
    if type == "":
        type = job['type']
    method_conf_name = "{}.{}".format(type, job['method'])
    method_conf = CONF_MAP[method_conf_name]()
    job[method_conf_name] = method_conf
    job['kmeans'] = _kmeans()
    return job
