import numpy as np

from anchor import anchor_tabular

from src.core.core import MODEL, ModelActions
from src.explanation.models import Explanation
from src.jobs.models import Job


def explain(anchor_exp: Explanation, training_df, test_df, explanation_target, prefix_target):
    job = Job.objects.filter(pk=anchor_exp.job.id)[0]

    explainer = anchor_tabular.AnchorTabularExplainer(
        [True, False],  # dataset.class_names
        job.encoding.features,  # = dataset.feature_names
        training_df.drop(['trace_id'], 1),  # dataset.data
        {
            item: list(range(max(training_df[item])))
            for item in job.encoding.features
        }
    )
    explainer.fit(
        training_df,  # dataset.train
        [True, False],  # dataset.labels_train
        test_df,  # dataset.validation
        [True, False]  # dataset.labels_validation
    )

    # show plot
    idx = 0
    np.random.seed(1)
    print('Prediction: ', explainer.class_names[
        MODEL[job.predictive_model.predictive_model][ModelActions.PREDICT.value](job, test_df)[0]])
    exp = explainer.explain_instance(test_df[idx],
                                     MODEL[job.predictive_model.predictive_model][ModelActions.PREDICT.value],
                                     threshold=0.95)
    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())
    """
	fit_anchor = np.where(np.all(dataset.test[:, exp.features()] == dataset.test[idx][exp.features()], axis=1))[0]
	print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(dataset.test.shape[0])))
	print('Anchor test precision: %.2f' % (
		np.mean(predict_fn(dataset.test[fit_anchor]) == predict_fn(dataset.test[idx].reshape(1, -1))))
		  )
	"""
    return dict(exp.names(), exp.precision(), exp.coverage())
