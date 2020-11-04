import os

import pandas as pd
import shap
from sklearn.externals import joblib

from src.encoding.common import retrieve_proper_encoder
from src.explanation.models import Explanation
from src.utils.file_service import create_unique_name


def explain(shap_exp: Explanation, training_df, test_df, explanation_target, prefix_target):
    job = shap_exp.job
    model = joblib.load(job.predictive_model.model_path)
    model = model[0]
    prefix_int = int(prefix_target.strip('/').split('_')[1])-1

    explainer = shap.TreeExplainer(model)
    target_df = test_df[test_df['trace_id'] == explanation_target].iloc[prefix_int]
    #if explanation_target is None:
    #    shap_values = explainer.shap_values(test_df.drop(['trace_id', 'label'], 1))
    #else:
    #    shap_values = explainer.shap_values(target_df.drop(['trace_id', 'label'], 0))

    shap_values = explainer.shap_values(target_df.drop(['trace_id', 'label'], 0))

    encoder = retrieve_proper_encoder(job)
    encoder.decode(test_df, job.encoding)
    target_df = test_df[test_df['trace_id'] == explanation_target].iloc[prefix_int]
    response = {explanation_target: [(target_df.keys()[index+1] + ' = ' + target_df[target_df.keys()[index+1]], shap_values[1][index]) for index in range(len(shap_values[1]))]}

    return response

    #explanation_target_int = test_df[test_df['trace_id'] == explanation_target].index.item() + \
    #                         training_df.drop(['trace_id', 'label'], 1).shape[0]

    #explanation_target_vector = test_df[test_df['trace_id'] == explanation_target].drop(['trace_id', 'label'], 1)
    #expected_value = explainer.expected_value[0] if len(explainer.expected_value) > 1 else explainer.expected_value
    #shap_value = shap_values[explanation_target_int, :] if hasattr(shap_values, "size") else shap_values[0][
    #                                                                                         explanation_target_int, :]
    #name = create_unique_name("temporal_shap.svg")
    #shap.force_plot(expected_value, shap_value, explanation_target_vector,
    #                show=False, matplotlib=True).savefig(name)
    #f = open(name, "r")
    #response = f.read()
    #os.remove(name)
    #if os.path.isfile(name.split('.svg')[0]):
    #    os.remove(name.split('.svg')[0])
