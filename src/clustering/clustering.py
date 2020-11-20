"""
clustering methods and functionalities
"""

import numpy as np
from pandas import Series, DataFrame
from sklearn.cluster import KMeans
from sklearn.externals import joblib

import src
from src.clustering.models import ClusteringMethods
from src.jobs.models import Job


class Clustering:
    """
    clustering related tasks, stores both the clustered data and the models trained on each cluster
    """

    def __init__(self, clustering: src.clustering.models.Clustering):
        """initializes the clustering class

        by default the number of clusters is set to 1, meaning no clustering

        :param job: job configuration

        """
        self.config = clustering.__getattribute__(
            clustering.clustering_method.lower()
        ).to_dict()
        self._choose_clusterer(
            clustering.__getattribute__(clustering.clustering_method.lower())
        )
        self.n_clusters = 1
        self.labels = [0]

    def fit(self, training_df: DataFrame) -> None:
        """clusters the input DataFrame

        :param training_df: training DataFrame

        """
        if hasattr(self.clusterer, 'fit'):
            self.clusterer.fit(training_df)
            self.labels = self.clusterer.labels_
            self.n_clusters = self.clusterer.n_clusters

    def predict(self, test_df: DataFrame) -> Series:  # TODO: check type hint
        """TODO: complete

        :param test_df: testing DataFrame
        :return: TODO: complete

        """
        if hasattr(self.clusterer, 'predict'):
            return self.clusterer.predict(
                test_df.drop([col for col in ['trace_id', 'label'] if col in test_df.columns], 1))
        return Series([0] * len(test_df))

    def cluster_data(self, input_df: DataFrame) -> dict:
        """clusters the input DataFrame

        :param input_df: input DataFrame
        :return: dictionary containing the clustered data
        """
        return {
            cluster: input_df.iloc[np.where(self.predict(input_df) == cluster)]
            for cluster in range(self.n_clusters)
        }

    def _choose_clusterer(self, clustering: src.clustering.models.Clustering):
        self.config.pop('clustering_method', None)
        if clustering.clustering_method == ClusteringMethods.KMEANS.value:
            self.clusterer = KMeans(**self.config)
        elif clustering.clustering_method == ClusteringMethods.NO_CLUSTER.value:
            self.clusterer = None
        else:
            raise ValueError("Unexpected clustering method {}".format(clustering.clustering_method))

    @classmethod
    def load_model(cls, job: Job):
        if job.clustering.clustering_method == ClusteringMethods.KMEANS.value:
            clusterer = joblib.load(job.clustering.model_path)
        elif job.clustering.clustering_method == ClusteringMethods.NO_CLUSTER.value:
            clusterer = Clustering(job.clustering)
        else:
            raise ValueError("Unexpected clustering method {}".format(job.clustering.clustering_method))
        return clusterer


def init_clusterer(clustering: Clustering, train_data: DataFrame):
    clusterer = Clustering(clustering)
    clusterer.fit(train_data.drop(['trace_id', 'label'], 1))
    return clusterer
