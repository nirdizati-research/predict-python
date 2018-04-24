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
LABELLING = 'labelling'

# Encoding options
ZERO_PADDING = 'zero_padding'

# classification/regressor method confs
CLASSIFICATION_RANDOM_FOREST = '{}.{}'.format(CLASSIFICATION, RANDOM_FOREST)
CLASSIFICATION_KNN = '{}.{}'.format(CLASSIFICATION, KNN)
CLASSIFICATION_DECISION_TREE = '{}.{}'.format(CLASSIFICATION, DECISION_TREE)
REGRESSION_RANDOM_FOREST = '{}.{}'.format(REGRESSION, RANDOM_FOREST)
REGRESSION_LASSO = '{}.{}'.format(REGRESSION, LASSO)
REGRESSION_LINEAR = '{}.{}'.format(REGRESSION, LINEAR)

all_configs = [CLASSIFICATION_RANDOM_FOREST, CLASSIFICATION_KNN, CLASSIFICATION_DECISION_TREE,
               REGRESSION_RANDOM_FOREST, REGRESSION_LASSO, REGRESSION_LINEAR]
