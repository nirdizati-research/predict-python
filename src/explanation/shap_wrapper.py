import os

import pandas as pd
import shap
from sklearn.externals import joblib

from src.encoding.common import retrieve_proper_encoder
from src.encoding.models import ValueEncodings
from src.explanation.models import Explanation
from src.utils.file_service import create_unique_name


def _init_explainer(model):
    return shap.TreeExplainer(model)


def _get_explanation(explainer, target_df):
    return explainer.shap_values(target_df)


def explain(shap_exp: Explanation, training_df, test_df, explanation_target, prefix_target):
    job = shap_exp.job
    model = joblib.load(job.predictive_model.model_path)
    model = model[0]
    prefix_int = int(prefix_target.strip('/').split('_')[1])-1

    explainer = _init_explainer(model)
    target_df = test_df[test_df['trace_id'] == explanation_target].iloc[prefix_int]
    #if explanation_target is None:
    #    shap_values = explainer.shap_values(test_df.drop(['trace_id', 'label'], 1))
    #else:
    #    shap_values = explainer.shap_values(target_df.drop(['trace_id', 'label'], 0))

    shap_values = _get_explanation(explainer, target_df.drop(['trace_id', 'label'], 0))

    encoder = retrieve_proper_encoder(job)
    encoder.decode(test_df, job.encoding)
    target_df = test_df[test_df['trace_id'] == explanation_target].iloc[prefix_int]
    response = {explanation_target: [(target_df.keys()[index+1] + ' = ' + target_df[target_df.keys()[index+1]],
                                      shap_values[1][index]) for index in range(len(shap_values[1]))]}

    return response


def _multi_trace_shap_temporal_stability(shap_exp: Explanation, training_df, test_df):
    #TODO: FIX FROM LIME_WRAPPER TO SHAP_WRAPPER
    model = joblib.load(shap_exp.predictive_model.model_path)[0]
    if len(model) > 1:
        raise NotImplementedError('Models with cluster-based approach are not yet supported')

    features = list(training_df.drop(['trace_id', 'label'], 1).columns.values)
    explainer = _init_explainer(model)

    #TODO: FILTER TO BE REMOVED BEFORE DEPLOY
    # test_df = test_df.head(100)

    exp = {}
    for trace_id in set(test_df['trace_id']):
        df = test_df[test_df['trace_id'] == trace_id].drop(['trace_id', 'label'], 1)
        df = df.reset_index(drop=True)

        if not any([feat.startswith('prefix_') for feat in features]) and len(df) == 1:
            exp[trace_id] = {
                'prefix_':
                    _get_explanation(
                        explainer,
                        row
                    )
                for position, row in df.iterrows()
            }
        else:
            exp[trace_id] = {
                row.index[max([feat for feat in range(len(features)) if
                               row.index[feat].startswith('prefix') and row[feat] != 0])]:
                    _get_explanation(
                        explainer,
                        row
                    )
                for position, row in df.iterrows()
            }

    encoder = retrieve_proper_encoder(shap_exp.job)

    encoder.decode(df=test_df, encoding=shap_exp.job.encoding)

    if shap_exp.job.encoding.value_encoding == ValueEncodings.BOOLEAN.value:
        for col in test_df:
            test_df[col] = test_df[col].apply(lambda x: 'False' if x == '0' else x)

    return {
        trace_id: {
            index: {
                el[0].split('=')[0]: {
                    'value': str(test_df[test_df['trace_id'] == trace_id].tail(1)[el[0].split('=')[0]].values[0]) if
                    el[0].split('=')[1] != '0' else '',
                    'importance': el[1]}
                for el in exp[trace_id][index]
            }
            for index in exp[trace_id]
        }
        for trace_id in set(test_df['trace_id'])
    }


def shap_temporal_stability(shap_exp: Explanation, training_df, test_df, explanation_target):
    if explanation_target is None:
        return _multi_trace_shap_temporal_stability(shap_exp, training_df, test_df)
    else:
        model = joblib.load(shap_exp.predictive_model.model_path)[0]

        features = list(training_df.drop(['trace_id', 'label'], 1).columns.values)
        explainer = _init_explainer(model)

        explanation_target_df = test_df[test_df['trace_id'] == explanation_target].drop(['trace_id', 'label'], 1)

        explanation_target_df = explanation_target_df.reset_index(drop=True)

        exp = {
            row.index[max([feat for feat in range(len(features)) if
                           row.index[feat].startswith('prefix') and row[feat] != 0])]: _get_explanation(
                explainer,
                row
            )
            for position, row in explanation_target_df.iterrows()
        }

        encoder = retrieve_proper_encoder(shap_exp.job)

        encoder.decode(df=explanation_target_df, encoding=shap_exp.job.encoding)

        return {
            explanation_target: {
                index: {explanation_target_df.keys()[idx]: {
                    'value': explanation_target_df.iloc[list(explanation_target_df.keys()).index('prefix_1')]
                    [explanation_target_df.keys()[idx]],
                    'importance': exp[index][1][idx]}
                    for idx in range(len(exp[index][1]))
                }
                for index in exp
            }
        }
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
