import numpy as np

from sklearn.cluster import DBSCAN

def dbscan(dataset, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(dataset.drop('label', 1), dataset['label'])

    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    return [dataset[labels == i] for i in xrange(n_clusters_)]
