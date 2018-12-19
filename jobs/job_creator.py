from copy import deepcopy

from core.constants import *
from core.default_configuration import CONF_MAP, _kmeans
from encoders.encoding_container import UP_TO, SIMPLE_INDEX
from jobs.models import Job, CREATED


def generate(split, payload, type=CLASSIFICATION):
    jobs = []

    for method in payload['config']['methods']:
        for clustering in payload['config']['clusterings']:
            for encMethod in payload['config']['encodings']:
                encoding = payload['config']['encoding']
                if encoding['generation_type'] == UP_TO:
                    for i in range(1, encoding['prefix_length'] + 1):
                        item = Job.objects.create(
                            split=split,
                            status=CREATED,
                            type=type,
                            config=deepcopy(create_config(payload, encMethod, clustering, method, i)))
                        jobs.append(item)
                else:
                    item = Job.objects.create(
                        split=split,
                        status=CREATED,
                        type=type,
                        config=create_config(payload, encMethod, clustering, method, encoding['prefix_length']))
                    jobs.append(item)

    return jobs


def generate_labelling(split, payload):
    jobs = []
    encoding = payload['config']['encoding']
    if encoding['generation_type'] == UP_TO:
        for i in range(1, encoding['prefix_length'] + 1):
            item = Job.objects.create(
                split=split,
                status=CREATED,
                type=LABELLING,
                config=deepcopy(create_config_labelling(payload, i)))
            jobs.append(item)
    else:
        item = Job.objects.create(
            split=split,
            status=CREATED,
            type=LABELLING,
            config=create_config_labelling(payload, encoding['prefix_length']))
        jobs.append(item)

    return jobs


def create_config(payload: dict, enc_method: str, clustering: str, method: str, prefix_length: int):
    """Turn lists to single values"""
    config = dict(payload['config'])
    del config['encodings']
    del config['clusterings']
    del config['methods']

    # Extract and merge configurations
    method_conf_name = "{}.{}".format(payload['type'], method)
    method_conf = {**CONF_MAP[method_conf_name](), **payload['config'].get(method_conf_name, dict())}
    # Remove configs that are not needed for this method
    for any_conf_name in all_configs:
        try:
            del config[any_conf_name]
        except KeyError:
            pass
    if clustering == KMEANS:
        config['kmeans'] = {**_kmeans(), **payload['config'].get('kmeans', dict())}
    elif 'kmeans' in config:
        del config['kmeans']
    config[method_conf_name] = method_conf
    config['clustering'] = clustering
    config['method'] = method
    # Encoding stuff rewrite
    config['encoding']['method'] = enc_method
    config['encoding']['prefix_length'] = prefix_length
    return config


def create_config_labelling(payload: dict, prefix_length: int):
    """For labelling job"""
    config = dict(payload['config'])

    # All methods are the same, so defaulting to SIMPLE_INDEX
    # Remove when encoding and labelling are separated
    config['encoding']['method'] = SIMPLE_INDEX
    config['encoding']['prefix_length'] = prefix_length
    return config
