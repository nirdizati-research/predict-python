from core.constants import *
from jobs.models import Job, CREATED


def generate(split, payload, type=CLASSIFICATION):
    jobs = []

    for encoding in payload['config']['encodings']:
        for clustering in payload['config']['clusterings']:
            for method in payload['config']['methods']:
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
    config[method_conf_name] = method_conf
    config['encoding'] = encoding
    config['clustering'] = clustering
    config['method'] = method
    config['prefix_length'] = prefix_length
    config['padding'] = payload['config']['prefix']['padding']
    return config


# Default configurations
def _classification_random_forest():
    return {
        'n_estimators': 10,
        'max_depth': None,
        'max_features': 'auto',
        'random_state': 21
    }


def _classification_knn():
    return {
        'n_neighbors': 5,
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
        'normalize': False
    }


# Map method config to a dict
CONF_MAP = {CLASSIFICATION_RANDOM_FOREST: _classification_random_forest, CLASSIFICATION_KNN: _classification_knn,
            CLASSIFICATION_DECISION_TREE: _classification_decision_tree,
            REGRESSION_RANDOM_FOREST: _regression_random_forest,
            REGRESSION_LASSO: _regression_lasso, REGRESSION_LINEAR: _regression_linear}
