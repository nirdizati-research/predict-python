from enum import Enum

from django.db import models

from src.common.models import CommonModel


class ClusteringMethods(Enum):
    KMEANS = 'kmeans'
    NO_CLUSTER = 'noCluster'


class Clustering(CommonModel):
    """Container of Classification to be shown in frontend"""
    # clustering_method = model
    @staticmethod
    def init(clustering: str = ClusteringMethods.NO_CLUSTER.value, configuration: dict = None):
        if clustering == ClusteringMethods.NO_CLUSTER.value:
            return NoClustering.objects.get_or_create(id=1)
        elif clustering == ClusteringMethods.KMEANS.value:
            from src.clustering.methods_default_config import clustering_kmeans #TODO fixme
            default_configuration = clustering_kmeans()
            return KMeans.objects.get_or_create(
                n_clusters=configuration.get('n_clusters', default_configuration['n_clusters']),
                init=configuration.get('init', default_configuration['init']),
                n_init=configuration.get('n_init', default_configuration['n_init']),
                max_iter=configuration.get('max_iter', default_configuration['max_iter']),
                tol=configuration.get('tol', default_configuration['tol']),
                precompute_distances=configuration.get('precompute_distances', default_configuration['precompute_distances']),
                random_state=configuration.get('random_state', default_configuration['random_state']),
                copy_x=configuration.get('copy_x', default_configuration['copy_x']),
                algorithm=configuration.get('algorithm', default_configuration['algorithm'])
            )
        else:
            raise ValueError('configuration ', clustering, 'not recognized')


class NoClustering(Clustering):
    pass


KMEANS_INIT_MAPPINGS = (
    ('k-means++', 'k-means++'),
    ('random', 'random')
)

KMEANS_PRECOMPUTE_DISTANCES_MAPPINGS = (
    (True, 'True'),
    (False, 'False'),
    ('auto', 'auto')
)

KMEANS_ALGORITHM_MAPPINGS = (
    ('auto', 'auto'),
    ('full', 'full'),
    ('elkan', 'elkan')
)


class KMeans(Clustering):
    n_clusters = models.PositiveIntegerField()
    init = models.CharField(choices=KMEANS_INIT_MAPPINGS, default='k-means++', max_length=20)
    n_init = models.PositiveIntegerField()
    max_iter = models.PositiveIntegerField()
    tol = models.FloatField()
    precompute_distances = models.CharField(choices=KMEANS_PRECOMPUTE_DISTANCES_MAPPINGS, default='auto', max_length=20)
    random_state = models.PositiveIntegerField()
    copy_x = models.BooleanField()
    algorithm = models.CharField(choices=KMEANS_ALGORITHM_MAPPINGS, default='auto', max_length=20)

    def to_dict(self) -> dict:
        return {
            'n_clusters': self.n_clusters,
            'init': self.init,
            'n_init': self.n_init,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'precompute_distances': self.precompute_distances,
            'random_state': self.random_state,
            'copy_x': self.copy_x,
            'algorithm': self.algorithm
        }
