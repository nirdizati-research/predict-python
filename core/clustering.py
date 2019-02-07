import numpy as np
from pandas import Series
from sklearn.cluster import KMeans

from core.constants import KMEANS, NO_CLUSTER

config = None
clusterer = None
labels = None
n_clusters = None


class Clustering:

    def __init__(self, job):
        self.config = job[KMEANS] if KMEANS in job else dict()
        self._choose_clusterer(job)
        self.n_clusters = 1
        self.labels = [0]

    def fit(self, df):
        if hasattr(self.clusterer, 'fit'):
            self.clusterer.fit(df)
            self.labels = self.clusterer.labels_
            self.n_clusters = self.clusterer.n_clusters

    def predict(self, df):
        if hasattr(self.clusterer, 'predict'):
            return self.clusterer.predict(df.drop([col for col in ['trace_id', 'label'] if col in df.columns], 1))
        else:
            return Series([0] * len(df))

    def cluster_data(self, df):
        return {
            cluster: df.iloc[np.where(self.predict(df) == cluster)]
            for cluster in range(self.n_clusters)
        }

    def _choose_clusterer(self, job):
        if job['clustering'] == KMEANS:
            self.clusterer = KMeans(**self.config)
        elif job['clustering'] == NO_CLUSTER:
            self.clusterer = None
        else:
            raise ValueError("Unexpected clustering method {}".format(job['clustering']))
