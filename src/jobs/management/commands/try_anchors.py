from anchor import anchor_tabular
from django.core.management.base import BaseCommand
from sklearn.externals import joblib

from src.core.core import get_encoded_logs
from src.jobs.models import Job


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        # get model
        TARGET_MODEL = 1090
        job = Job.objects.filter(pk=TARGET_MODEL)[0]
        model = joblib.load(job.predictive_model.model_path)
        model = model[0]

        # load data
        training_df, test_df = get_encoded_logs(job)

        # get radom point in evaluation set
        EXPLANATION_TARGET = 1

        # get the actual explanation
        explainer = anchor_tabular.AnchorTabularExplainer(
            dataset.class_names,
            dataset.feature_names,
            dataset.data,
            dataset.categorical_names
        )
        explainer.fit(
            dataset.train,
            dataset.labels_train,
            dataset.validation,
            dataset.labels_validation
        )

        # show plot
        idx = 0
        np.random.seed(1)
        print('Prediction: ', explainer.class_names[predict_fn(dataset.test[idx].reshape(1, -1))[0]])
        exp = explainer.explain_instance(dataset.test[idx], c.predict, threshold=0.95)
        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print('Coverage: %.2f' % exp.coverage())

        fit_anchor = np.where(np.all(dataset.test[:, exp.features()] == dataset.test[idx][exp.features()], axis=1))[0]
        print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(dataset.test.shape[0])))
        print('Anchor test precision: %.2f' % (
            np.mean(predict_fn(dataset.test[fit_anchor]) == predict_fn(dataset.test[idx].reshape(1, -1))))
              )

        print('done')
