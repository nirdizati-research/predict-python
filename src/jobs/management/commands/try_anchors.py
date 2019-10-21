from anchor import anchor_tabular
from django.core.management.base import BaseCommand
from sklearn.externals import joblib

from src.core.core import get_encoded_logs
from src.jobs.models import Job

import numpy as np


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        # get model
        TARGET_MODEL = 1
        job = Job.objects.filter(pk=TARGET_MODEL)[0]
        model = joblib.load(job.predictive_model.model_path)
        model = model[0]

        # load data
        training_df, test_df = get_encoded_logs(job)

        # get radom point in evaluation set
        EXPLANATION_TARGET = 1

        # get the actual explanation
        job.encoding.features.remove('label')
        explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=[True, False],
            feature_names=job.encoding.features,
            data=training_df.drop(['trace_id', 'label'], 1).T,
            categorical_names={
                job.encoding.features.index(item): list(range(max(training_df[item])))
                for item in job.encoding.features
            }
        )
        explainer.fit(
            training_df.drop(['trace_id', 'label'], 1).as_matrix(),
            [True, False],
            test_df.drop(['trace_id', 'label'], 1).as_matrix(),
            [True, False]
        )

        model_fn = lambda x: model.predict(x)

        # show plot
        idx = 0
        np.random.seed(1)
        print('Prediction: ', explainer.class_names[model_fn(test_df.drop(['trace_id', 'label'], 1).as_matrix()[idx].reshape(1, -1))[0]])
        exp = explainer.explain_instance(test_df.drop(['trace_id', 'label'], 1).as_matrix()[idx], model_fn, threshold=0.95)
        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print('Coverage: %.2f' % exp.coverage())

        fit_anchor = np.where(np.all(test_df.drop(['trace_id', 'label'], 1)[:, exp.features()] == test_df.drop(['trace_id', 'label'], 1).as_matrix()[idx][exp.features()], axis=1))[0]
        print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(test_df.drop(['trace_id', 'label'], 1).shape[0])))
        # print('Anchor test precision: %.2f' % (
        #     np.mean(predict_fn(test_df.drop(['trace_id', 'label'], 1)[fit_anchor]) == predict_fn(test_df.drop(['trace_id', 'label'], 1).as_matrix()[idx].reshape(1, -1))))
        #     np.mean(predict_fn(test_df.drop(['trace_id', 'label'], 1)[fit_anchor]) == predict_fn(test_df.drop(['trace_id', 'label'], 1).as_matrix()[idx].reshape(1, -1))))
        #       )

        print('done')
