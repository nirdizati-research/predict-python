import numpy as np
from pandas import Series
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from predModels.models import PredModels, ModelSplit


class Clustering:
    KMEANS = 'kmeans'
    NO_CLUSTER = 'noCluster'

    def __init__(self, job):
        self.config = job[self.KMEANS] if self.KMEANS in job else dict()
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

    def _choose_clusterer(self, job): #TODO this will change when using more than one type of cluster
        if job['clustering'] == self.KMEANS:
            self.clusterer = KMeans(**self.config)
        elif job['clustering'] == self.NO_CLUSTER:
            self.clusterer = None
        else:
            raise ValueError("Unexpected clustering method {}".format(job['clustering']))

    @classmethod
    def load_model(cls, job):
        if job['clustering'] == cls.KMEANS:
            classifier = PredModels.objects.filter(id=job['incremental_train']['base_model'])
            assert len(classifier) == 1  # asserting that the used id is unique
            classifier_details = classifier[0]
            classifier = ModelSplit.objects.filter(id=classifier_details.split_id)
            assert len(classifier) == 1
            classifier = classifier[0]
            #TODO this is a bad workaround
            clusterer = joblib.load(classifier.model_path[:11] + classifier.model_path[11:].replace('model', 'clusterer'))
        elif job['clustering'] == cls.NO_CLUSTER:
            clusterer = Clustering(job)
        else:
            raise ValueError("Unexpected clustering method {}".format(job['clustering']))
        return clusterer
