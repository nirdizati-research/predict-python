from django.db import models


class Clustering(models.Model):
    """Container of Classification to be shown in frontend"""

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
