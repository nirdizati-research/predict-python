import shap

from src.encoding.common import retrieve_proper_encoder
from src.explanation.models import Explanation
from sklearn.externals import joblib
import os
import pandas as pd


def explain(shap_exp: Explanation, training_df, test_df, explanation_target):
    job = shap_exp.job
    job
    model = joblib.load(job.predictive_model.model_path)
    model = model[0]
    shap.initjs()

    explainer = shap.TreeExplainer(model)
    merged_df = pd.concat([training_df, test_df])
    shap_values = explainer.shap_values(merged_df.drop(['trace_id', 'label'], 1))

    encoder = retrieve_proper_encoder(job)
    encoder.decode(merged_df, job.encoding)
    encoder.decode(test_df, job.encoding)

    explanation_target_int = merged_df[merged_df['trace_id'] == explanation_target].index.item() + \
                             training_df.drop(['trace_id', 'label'], 1).shape[0]

    explanation_target_vector = test_df[test_df['trace_id'] == explanation_target].drop(['trace_id', 'label'], 1)
    expected_value = explainer.expected_value[0] if explainer.expected_value.size > 1 else explainer.expected_value
    shap_value = shap_values[explanation_target_int, :] if hasattr(shap_values,"size") else shap_values[0][
                                                                                          explanation_target_int, :]
    shap.force_plot(expected_value, shap_value, explanation_target_vector,
                    show=False, matplotlib=True).savefig("temporal_shap.svg")
    f = open("temporal_shap.svg", "r")
    response = f.read()
    os.remove("temporal_shap.svg")
    return response
