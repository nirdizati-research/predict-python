def hyperparameter_optimization_hyperopt():
    from src.hyperparameter_optimization.models import HyperOptAlgorithms, HyperOptLosses

    return {
        'max_evaluations': 2,
        'performance_metric': HyperOptLosses.ACC.value,
        'algorithm_type': HyperOptAlgorithms.RANDOM_SEARCH.value
    }
