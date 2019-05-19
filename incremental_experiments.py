import json

import requests

print("Starting experiments")

def create_payload(
    SPLIT=1,
    ENCODINGS=["simpleIndex"],
    ENCODING={"padding": "zero_padding","generation_type": "all_in_one","prefix_length": 5},
    LABELING={"type": "attribute_string","attribute_name": "creator","threshold_type": "threshold_mean","threshold": 0,"add_remaining_time": False,"add_elapsed_time": False,"add_executed_events": False,"add_resources_used": False,"add_new_traces": False},
    CLUSTERING=["noCluster"],
    CLASSIFICATION=["multinomialNB"],
    HYPERPARAMETER_OPTIMIZATION={"type": 'hyperopt',"max_evals": 10,"performance_metric": "auc"},
    INCREMENTAL_TRAIN={"base_model": None}):

    CONFIG = {
        "clusterings": CLUSTERING,
        "labelling": LABELING,
        "encodings": ENCODINGS,
        "encoding": ENCODING,
        "hyperparameter_optimizer": HYPERPARAMETER_OPTIMIZATION,
        "methods": CLASSIFICATION,
        "incremental_train": INCREMENTAL_TRAIN,
        "create_models": "True",
        "kmeans": {},
        "classification.knn": {},
        "classification.randomForest": {},
        "classification.decisionTree": {},
        "classification.xgboost": {},
        "classification.multinomialNB": {},
        "classification.hoeffdingTree": {},
        "classification.adaptiveTree": {},
        "classification.SGDClassifier": {},
        "classification.perceptron": {},
        "classification.nn": {},
        "regression.randomForest": {},
        "regression.lasso": {},
        "regression.linear": {},
        "regression.xgboost": {},
        "regression.nn": {},
        "timeSeriesPrediction.rnn": {}}

    return {"type": "classification", "split_id": SPLIT, "config": CONFIG}


def send_request(PAYLOAD):
    SERVER_NAME = "localhost"
    SERVER_PORT = '8000'
    headers = {'Content-type': 'application/json'}
    r = requests.post('http://' + SERVER_NAME + ':' + SERVER_PORT + '/jobs/multiple', json=PAYLOAD, headers=headers)
    return json.loads(r.text)


json_data = send_request(PAYLOAD=create_payload())

INCREMENTAL_TRAIN = {"base_model": json_data[0]['id']}
HYPERPARAMETER_OPTIMIZATION = {"type": 'none'}

json_data = send_request(PAYLOAD=create_payload(SPLIT=1, HYPERPARAMETER_OPTIMIZATION=HYPERPARAMETER_OPTIMIZATION, INCREMENTAL_TRAIN=INCREMENTAL_TRAIN))

print("End of the experiemnts")

