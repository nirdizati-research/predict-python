import os

import pandas as pd
import shap
from sklearn.externals import joblib

from src.encoding.common import retrieve_proper_encoder
from src.explanation.models import Explanation
from src.utils.file_service import create_unique_name


def explain(shap_exp: Explanation, training_df, test_df, explanation_target):
    job = shap_exp.job
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
    shap_value = shap_values[explanation_target_int, :] if hasattr(shap_values, "size") else shap_values[0][
                                                                                             explanation_target_int, :]
    name = create_unique_name("temporal_shap.svg")
    shap.force_plot(expected_value, shap_value, explanation_target_vector,
                    show=False, matplotlib=True).savefig(name)
    f = open(name, "r")
    response = f.read()
    os.remove(name)
    os.remove(name.split('.svg')[0])
    return response
