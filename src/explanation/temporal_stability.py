from pandas import DataFrame

from src.clustering.models import ClusteringMethods
from src.core.core import MODEL, ModelActions
from src.encoding.common import retrieve_proper_encoder
from src.explanation.models import Explanation
from src.predictive_model.models import PredictiveModels


def _multi_trace_temporal_stability(temporal_stability_exp: Explanation, training_df, test_df):
    if temporal_stability_exp.job.clustering.clustering_method != ClusteringMethods.NO_CLUSTER.value:
        raise NotImplementedError('Models with cluster-based approach are not yet supported')

    test_df['predicted'] = MODEL[PredictiveModels.CLASSIFICATION.value][ModelActions.PREDICT.value](temporal_stability_exp.job, test_df)

    encoder = retrieve_proper_encoder(temporal_stability_exp.job)

    encoder.decode(df=test_df, encoding=temporal_stability_exp.job.encoding)

    temp_df = DataFrame()
    temp_df['label'] = test_df['predicted']
    encoder.decode(df=temp_df, encoding=temporal_stability_exp.job.encoding)
    test_df['predicted'] = temp_df['label']

    exp_list = {}
    for trace_id in set(test_df['trace_id']):
        df = test_df[test_df['trace_id'] == trace_id].drop(['trace_id', 'label'], 1)
        exp = list(df['predicted'])
        last_row = df.tail(1)
        exp_list_1 = [(feat, str(last_row[feat].values[0])) for feat in last_row]
        exp_list[trace_id] = {
            exp_list_1[index][0]: {'value': exp_list_1[index][1], 'predicted': exp[index]}
            for index in range(len(exp))
        }

    return exp_list


def temporal_stability(temporal_stability_exp: Explanation, training_df, test_df, explanation_target):
    if temporal_stability_exp.job.clustering.clustering_method != ClusteringMethods.NO_CLUSTER.value:
        raise NotImplementedError('Models with cluster-based approach are not yet supported')

    if explanation_target is None:
        return _multi_trace_temporal_stability(temporal_stability_exp, training_df, test_df)
    else:
        explanation_target_df = test_df[test_df['trace_id'] == explanation_target]

        exp = MODEL[PredictiveModels.CLASSIFICATION.value][ModelActions.PREDICT.value](temporal_stability_exp.job, explanation_target_df)

        encoder = retrieve_proper_encoder(temporal_stability_exp.job)

        encoder.decode(df=explanation_target_df, encoding=temporal_stability_exp.job.encoding)

        last_row = explanation_target_df.drop(['trace_id', 'label'], 1).tail(1)
        exp_list_1 = [(feat, str(last_row[feat].values[0])) for feat in last_row]
        exp = exp.tolist()
        temp_df = DataFrame()
        temp_df['label'] = exp
        encoder.decode(df=temp_df, encoding=temporal_stability_exp.job.encoding)
        exp = list(temp_df['label'])
        return {
            explanation_target: {
                    exp_list_1[index][0]: {'value': exp_list_1[index][1], 'predicted': exp[index]}
                    for index in range(len(exp))
            }
        }
