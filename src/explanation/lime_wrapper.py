import lime
from sklearn.externals import joblib

from src.encoding.common import retrieve_proper_encoder
from src.encoding.models import ValueEncodings
from src.explanation.models import Explanation


def _init_explainer(df, features, columns, mode):
    return lime.lime_tabular.LimeTabularExplainer(
        df,
        feature_names=features,
        categorical_features=[i for i in range(len(columns))],
        verbose=True,
        mode=mode,
    )


def _get_explanation(explainer, explanation_target_vector, model, features):
    return explainer.explain_instance(
        explanation_target_vector,
        # TODO probably the opposite would be way less computationally intesive
        model[0].predict if explainer.mode == 'regression' else model[0].predict_proba,  # TODO if we have clustering this is using only first model
        num_features=len(features)
    )

def explain(lime_exp: Explanation, training_df, test_df, explanation_target=1):
    model = joblib.load(lime_exp.predictive_model.model_path)
    if len(model) > 1:
        raise NotImplementedError('Models with cluster-based approach are not yet supported')

    # get the actual explanation
    features = list(training_df.drop(['trace_id', 'label'], 1).columns.values)
    explainer = _init_explainer(
        df=training_df.drop(['trace_id', 'label'], 1).as_matrix(),
        features=features,
        columns=list(training_df.drop(['trace_id', 'label'], 1).columns.values),
        mode=getModeType(model[0])
    )

    explanation_target_vector = test_df[test_df['trace_id'] == explanation_target].drop(['trace_id', 'label'], 1).tail(
        1).squeeze()
    exp = _get_explanation(
        explainer=explainer,
        explanation_target_vector=explanation_target_vector,
        model=model,
        features=features
    )

    # show plot
    # exp.show_in_notebook(show_table=True)
    # exp.as_pyplot_figure().show()
    # exp.save_to_file('/tmp/oi.html')

    # alternative visualisation
    # exp.as_map()

    encoder = retrieve_proper_encoder(lime_exp.job)

    exp_list = exp.as_list()

    explanation_target_df = explanation_target_vector.to_frame().T
    encoder.decode(df=explanation_target_df, encoding=lime_exp.job.encoding)

    return {
        e[0].split('=')[0]:
            (str(explanation_target_df[e[0].split('=')[0]].values[0]), e[1])
        for e in exp_list
    }


def _multi_trace_lime_temporal_stability(lime_exp: Explanation, training_df, test_df):
    model = joblib.load(lime_exp.predictive_model.model_path)
    if len(model) > 1:
        raise NotImplementedError('Models with cluster-based approach are not yet supported')

    features = list(training_df.drop(['trace_id', 'label'], 1).columns.values)
    explainer = _init_explainer(
        df=training_df.drop(['trace_id', 'label'], 1).as_matrix(),
        features=features,
        columns=list(training_df.drop(['trace_id', 'label'], 1).columns.values),
        mode=getModeType(model[0])
    )

    #TODO: FILTER TO BE REMOVED BEFORE DEPLOY
    # test_df = test_df.head(100)

    exp = {}
    for trace_id in set(test_df['trace_id']):
        df = test_df[test_df['trace_id'] == trace_id].drop(['trace_id', 'label'], 1)
        # filterded_df = pd.DataFrame()
        # try:
        #     filterded_df = filterded_df.append(df.head(30).tail(1))
        # except:
        #     pass
        # try:
        #     filterded_df = filterded_df.append(df.head(60).tail(1))
        # except:
        #     pass
        # try:
        #     filterded_df = filterded_df.append(df.head(90).tail(1))
        # except:
        #     pass
        #
        # df = filterded_df
        df = df.reset_index(drop=True)

        if not any([feat.startswith('prefix_') for feat in features]) and len(df) == 1:
            exp[trace_id] = {
                'prefix_':
                    _get_explanation(
                        explainer,
                        explanation_target_vector=row,
                        model=model,
                        features=features
                    ).as_list()
                for position, row in df.iterrows()
            }
        else:
            exp[trace_id] = {
                row.index[max([feat for feat in range(len(features)) if
                               row.index[feat].startswith('prefix') and row[feat] != 0])]:
                    _get_explanation(
                        explainer,
                        explanation_target_vector=row,
                        model=model,
                        features=features
                    ).as_list()
                for position, row in df.iterrows()
            }

        # exp[trace_id] = {
        #     row.index[max([ feat for feat in range(len(features)) if row.index[feat].startswith('prefix') and row[feat] != 0 ])]:
        #         _get_explanation(
        #         explainer,
        #         explanation_target_vector=row,
        #         model=model,
        #         features=features
        #     ).as_list()
        #     for position, row in df.iterrows()
        # }

    encoder = retrieve_proper_encoder(lime_exp.job)

    encoder.decode(df=test_df, encoding=lime_exp.job.encoding)

    if lime_exp.job.encoding.value_encoding == ValueEncodings.BOOLEAN.value:
        for col in test_df:
            test_df[col] = test_df[col].apply(lambda x: 'False' if x == '0' else x)

    return {
        trace_id: {
            index: {
                el[0].split('=')[0]: {
                    'value': str(test_df[test_df['trace_id'] == trace_id].tail(1)[el[0].split('=')[0]].values[0]) if el[0].split('=')[1] != '0' else '',
                    'importance': el[1]}
                for el in exp[trace_id][index]
            }
            for index in exp[trace_id]
        }
        for trace_id in set(test_df['trace_id'])
    }


def lime_temporal_stability(lime_exp: Explanation, training_df, test_df, explanation_target):
    if explanation_target is None:
        return _multi_trace_lime_temporal_stability(lime_exp, training_df, test_df)
    else:
        model = joblib.load(lime_exp.predictive_model.model_path)
        if len(model) > 1:
            raise NotImplementedError('Models with cluster-based approach are not yet supported')

        features = list(training_df.drop(['trace_id', 'label'], 1).columns.values)
        explainer = _init_explainer(
            df=training_df.drop(['trace_id', 'label'], 1).as_matrix(),
            features=features,
            columns=list(training_df.drop(['trace_id', 'label'], 1).columns.values),
            mode=getModeType(model[0])
        )

        explanation_target_df = test_df[test_df['trace_id'] == explanation_target].drop(['trace_id', 'label'], 1)

        explanation_target_df = explanation_target_df.reset_index(drop=True)

        exp = {
            row.index[max([ feat for feat in range(len(features)) if row.index[feat].startswith('prefix') and row[feat] != 0 ])]: _get_explanation(
                explainer,
                explanation_target_vector=row,
                model=model,
                features=features
            ).as_list()
            for position, row in explanation_target_df.iterrows()
        }

        encoder = retrieve_proper_encoder(lime_exp.job)

        encoder.decode(df=explanation_target_df, encoding=lime_exp.job.encoding)

        return {
            explanation_target: {
                index: {el[0].split('=')[0]: {
                    'value': explanation_target_df.tail(1)[el[0].split('=')[0]].values[0] if el[0].split('=')[
                                                                                                 1] != '0' else '',
                    'importance': el[1]}
                    for el in exp[index]
                }
                for index in exp
            }
        }


def getModeType(model):
    return 'regression' if model._estimator_type == 'regressor' else 'classification'
