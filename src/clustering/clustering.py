"""
clustering methods and functionalities
"""

import numpy as np
from pandas import Series, DataFrame
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from src.jobs.models import Job


class Clustering:
    """
    clustering related tasks, stores both the clustered data and the models trained on each cluster
    """

    KMEANS = 'kmeans'
    NO_CLUSTER = 'noCluster'

    def __init__(self, job: Job):
        """initializes the clustering class

        by default the number of clusters is set to 1, meaning no clustering

        :param job: job configuration

        """
        self.config = job[self.KMEANS] if self.KMEANS in job else dict()
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
        return {
            cluster: input_df.iloc[np.where(self.predict(input_df) == cluster)]
            for cluster in range(self.n_clusters)
        }

    def _choose_clusterer(self, job):  # TODO this will change when using more than one type of cluster
        if job['clustering'] == self.KMEANS:
            #TODO: retrieve entry from db or create new one
            # Clustering.objects.get_or_create(split=, encoding=, labelling=, config= )
            self.clusterer = KMeans(**self.config)
        elif job['clustering'] == self.NO_CLUSTER:
            self.clusterer = None
        else:
            raise ValueError("Unexpected clustering method {}".format(job['clustering']))

    @classmethod
    def load_model(cls, job):
        if job['clustering'] == cls.KMEANS:
            pass
            # TODO fixme
            # classifier = PredModels.objects.filter(id=job['incremental_train']['base_model'])
            # assert len(classifier) == 1  # asserting that the used id is unique
            # classifier_details = classifier[0]
            # classifier = ModelSplit.objects.filter(id=classifier_details.split_id)
            # assert len(classifier) == 1
            # classifier = classifier[0]
            # TODO this is a bad workaround
            # clusterer = joblib.load(
            #     classifier.model_path[:11] + classifier.model_path[11:].replace('predictive_model', 'clusterer'))
        elif job['clustering'] == cls.NO_CLUSTER:
            clusterer = Clustering(job)
        else:
            raise ValueError("Unexpected clustering method {}".format(job['clustering']))
        return clusterer
