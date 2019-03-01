def clustering_kmeans():
    return {
        'n_clusters': 8,
        'init': 'k-means++',
        'n_init': 10,
        'max_iter': 300,
        'tol': 1e-4,
        'precompute_distances': 'auto',
        'random_state': None,
        'copy_x': None,
        'algorithm': None
    }
