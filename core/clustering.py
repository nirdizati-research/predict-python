"""
clustering methods and functionalities
"""

import numpy as np
from pandas import Series, DataFrame
from sklearn.cluster import KMeans

from core.constants import KMEANS, NO_CLUSTER

config = None
clusterer = None
labels = None
n_clusters = None


class Clustering:
    """
    clustering related tasks, stores both the clustered data and the models trained on each cluster
    """

    def __init__(self, job: dict):
        """initializes the clustering class

        by default the number of clusters is set to 1, meaning no clustering

        :param job: job configuration

        """
        self.config = job[KMEANS] if KMEANS in job else dict()
        self._choose_clusterer(job)
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
        return {cluster: input_df.iloc[np.where(self.predict(input_df) == cluster)] for cluster in
                range(self.n_clusters)}

    def _choose_clusterer(self, job: dict) -> None:
        if job['clustering'] == KMEANS:
            self.clusterer = KMeans(**self.config)
        elif job['clustering'] == NO_CLUSTER:
            self.clusterer = None
        else:
            raise ValueError("Unexpected clustering method {}".format(job['clustering']))
