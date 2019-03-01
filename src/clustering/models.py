from django.db import models

from src.core.constants import KMEANS, NO_CLUSTER
from src.core.default_configuration import clustering_kmeans


class Clustering(models.Model):
    """Container of Classification to be shown in frontend"""

    @staticmethod
    def init(clustering: str = NO_CLUSTER, configuration: dict = None):
        if clustering == NO_CLUSTER:
            return NoClustering.objects.get_or_create(id=1)
        elif clustering == KMEANS:
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

    def to_dict(self) -> dict:
        return {}


class NoClustering(Clustering):

    def to_dict(self) -> dict:
        return {}


KMEANS_INIT = (
    ('k-means++', 'k-means++'),
    ('random', 'random')
)

KMEANS_PRECOMPUTE_DISTANCES = (
    (True, 'True'),
    (False, 'False'),
    ('auto', 'auto')
)

KMEANS_ALGORITHM = (
    ('auto', 'auto'),
    ('full', 'full'),
    ('elkan', 'elkan')
)


class KMeans(Clustering):
    n_clusters = models.PositiveIntegerField()
    init = models.CharField(choices=KMEANS_INIT, default='k-means++', max_length=20)
    n_init = models.PositiveIntegerField()
    max_iter = models.PositiveIntegerField()
    tol = models.FloatField()
    precompute_distances = models.CharField(choices=KMEANS_PRECOMPUTE_DISTANCES, default='auto', max_length=20)
    random_state = models.PositiveIntegerField()
    copy_x = models.BooleanField()
    algorithm = models.CharField(choices=KMEANS_ALGORITHM, default='auto', max_length=20)

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
