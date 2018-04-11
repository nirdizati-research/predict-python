"""Python has no real constants, will still try my best"""

# Encoding methods
SIMPLE_INDEX = 'simpleIndex'
BOOLEAN = 'boolean'
FREQUENCY = 'frequency'
COMPLEX = 'complex'
LAST_PAYLOAD = 'lastPayload'

# Classification methods
KNN = 'knn'
RANDOM_FOREST = 'randomForest'
DECISION_TREE = 'decisionTree'

# Regression methods
LINEAR = 'linear'
LASSO = 'lasso'

# Clustering
KMEANS = 'kmeans'
NO_CLUSTER = 'noCluster'

# Job types
CLASSIFICATION = 'classification'
REGRESSION = 'regression'
NEXT_ACTIVITY = 'nextActivity'

# classification/regressor method confs
CLASSIFICATION_RANDOM_FOREST = '{}.{}'.format(CLASSIFICATION, RANDOM_FOREST)
CLASSIFICATION_KNN = '{}.{}'.format(CLASSIFICATION, KNN)
CLASSIFICATION_DECISION_TREE = '{}.{}'.format(CLASSIFICATION, DECISION_TREE)
NEXT_ACTIVITY_RANDOM_FOREST = '{}.{}'.format(NEXT_ACTIVITY, RANDOM_FOREST)
NEXT_ACTIVITY_KNN = '{}.{}'.format(NEXT_ACTIVITY, KNN)
NEXT_ACTIVITY_DECISION_TREE = '{}.{}'.format(NEXT_ACTIVITY, DECISION_TREE)
REGRESSION_RANDOM_FOREST = '{}.{}'.format(REGRESSION, RANDOM_FOREST)
REGRESSION_LASSO = '{}.{}'.format(REGRESSION, LASSO)
REGRESSION_LINEAR = '{}.{}'.format(REGRESSION, LINEAR)

all_configs = [CLASSIFICATION_RANDOM_FOREST, CLASSIFICATION_KNN, CLASSIFICATION_DECISION_TREE,
               NEXT_ACTIVITY_RANDOM_FOREST, NEXT_ACTIVITY_KNN, NEXT_ACTIVITY_DECISION_TREE, REGRESSION_RANDOM_FOREST,
               REGRESSION_LASSO, REGRESSION_LINEAR]
