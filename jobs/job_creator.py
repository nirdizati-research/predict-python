from core.constants import *
from jobs.models import Job, CREATED


def generate(split, payload, type=CLASSIFICATION):
    jobs = []

    for method in payload['config']['methods']:
        for clustering in payload['config']['clusterings']:
            for encoding in payload['config']['encodings']:
                prefix = payload['config']['prefix']
                if prefix['type'] == 'up_to':
                    for i in range(1, prefix['prefix_length'] + 1):
                        item = Job.objects.create(
                            split=split,
                            status=CREATED,
                            type=type,
                            config=create_config(payload, encoding, clustering, method, i))
                        jobs.append(item)
                else:
                    item = Job.objects.create(
                        split=split,
                        status=CREATED,
                        type=type,
                        config=create_config(payload, encoding, clustering, method, prefix['prefix_length']))
                    jobs.append(item)

    return jobs


def generate_labelling(split, payload):
    jobs = []
    prefix = payload['config']['prefix']
    if prefix['type'] == 'up_to':
        for i in range(1, prefix['prefix_length'] + 1):
            item = Job.objects.create(
                split=split,
                status=CREATED,
                type=LABELLING,
                config=create_config_labelling(payload, i))
            jobs.append(item)
    else:
        item = Job.objects.create(
            split=split,
            status=CREATED,
            type=LABELLING,
            config=create_config_labelling(payload, prefix['prefix_length']))
        jobs.append(item)

    return jobs


def create_config(payload: dict, encoding: str, clustering: str, method: str, prefix_length: int):
    """Turn lists to single values"""
    config = dict(payload['config'])
    del config['encodings']
    del config['clusterings']
    del config['methods']
    del config['prefix']

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
    config['encoding'] = encoding
    config['clustering'] = clustering
    config['method'] = method
    config['prefix_length'] = prefix_length
    config['padding'] = payload['config']['prefix']['padding']
    return config


def create_config_labelling(payload: dict, prefix_length: int):
    """For labelling job"""
    config = dict(payload['config'])
    del config['prefix']

    # All methods are the same, so defaulting to SIMPLE_INDEX
    # Remove when encoding and labelling are separated
    config['encoding'] = SIMPLE_INDEX
    config['prefix_length'] = prefix_length
    config['padding'] = payload['config']['prefix']['padding']
    return config


# Default configurations
def _classification_random_forest():
    return {
        'n_estimators': 10,
        'max_depth': None,
        'max_features': 'auto',
        'n_jobs': -1,
        'random_state': 21
    }


def _classification_knn():
    return {
        'n_neighbors': 5,
        'n_jobs': -1,
        'weights': 'uniform'
    }


def _classification_decision_tree():
    return {
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 21
    }


def _regression_random_forest():
    return {
        'n_estimators': 10,
        'max_depth': None,
        'max_features': 'auto',
        'n_jobs': -1,
        'random_state': 21
    }


def _regression_lasso():
    return {
        'alpha': 1.0,
        'fit_intercept': True,
        'normalize': False,
        'random_state': 21
    }


def _regression_linear():
    return {
        'fit_intercept': True,
        'n_jobs': -1,
        'normalize': False
    }


def _kmeans():
    return {
        'n_clusters': 3,
        'max_iter': 300,
        'n_jobs': -1,
        'algorithm': 'auto',
        'random_state': 21
    }


# Map method config to a dict
CONF_MAP = {CLASSIFICATION_RANDOM_FOREST: _classification_random_forest, CLASSIFICATION_KNN: _classification_knn,
            CLASSIFICATION_DECISION_TREE: _classification_decision_tree,
            REGRESSION_RANDOM_FOREST: _regression_random_forest,
            REGRESSION_LASSO: _regression_lasso, REGRESSION_LINEAR: _regression_linear}
