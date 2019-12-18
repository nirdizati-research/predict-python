def clustering_kmeans():
    return {
        'n_clusters': 3,
        'init': 'k-means++',
        'n_init': 10,
        'max_iter': 300,
        'tol': 1e-4,
        'precompute_distances': 'auto',
        'random_state': 1,  # use a fixed int to make the init deterministic
        'copy_x': None,
        'algorithm': 'full'
    }
