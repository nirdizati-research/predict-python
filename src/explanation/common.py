import lime
import lime.lime_tabular
import numpy as np
import shap
from anchor import anchor_tabular

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from sklearn.externals import joblib

from src.core.core import get_encoded_logs, ModelActions, MODEL
from src.encoding.encoder import Encoder
from src.jobs.models import Job
from src.split.splitting import get_train_test_log


@api_view(['GET'])
def get_lime(request, pk):  # TODO: changed self to request, check if correct or not
    # get model
    TARGET_MODEL = 1090
    job = Job.objects.filter(pk=pk)[0]
    model = joblib.load(job.predictive_model.model_path)

    # load data
    training_df, test_df = get_encoded_logs(job)

    # get random point in evaluation set
    EXPLANATION_TARGET = 1

    # get the actual explanation
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_df.drop(['trace_id', 'label'], 1).as_matrix(),
        feature_names=list(training_df.drop(['trace_id', 'label'], 1).columns.values),
        categorical_features=[i for i in range(len(list(training_df.drop(['trace_id', 'label'], 1).columns.values)))],
        verbose=True,
        mode='classification',
    )
    exp = explainer.explain_instance(
        test_df.drop(['trace_id', 'label'], 1).iloc[EXPLANATION_TARGET],
        # TODO probably the opposite would be way less computationally intesive
        model[0].predict_proba,
        num_features=5
    )
    exp.as_list()

    # show plot
    exp.show_in_notebook(show_table=True)
    exp.as_pyplot_figure().show()
    # exp.save_to_file('/tmp/oi.html')

    print('done')
    return Response(exp.as_map(), status=200)


@api_view(['GET'])
def get_shap(request, pk):  # TODO: changed self to request, check if correct or not
    job = Job.objects.filter(pk=pk)[0]
    model = joblib.load(job.predictive_model.model_path)
    model = model[0]

    # load data
    training_df, test_df = get_encoded_logs(job)

    # get random point in evaluation set
    EXPLANATION_TARGET = 1

    # get the actual explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(training_df)

    # show plot
    shap.summary_plot(shap_values, training_df)
    shap.summary_plot(shap_values, training_df, plot_type="bar")

    # TODO not yet working
    shap.force_plot(explainer.expected_value, shap_values[EXPLANATION_TARGET, :],
                    training_df.iloc[EXPLANATION_TARGET, :])
    shap.force_plot(explainer.expected_value, shap_values, training_df)
    shap.dependence_plot("RM", shap_values, training_df)
    shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], test_df.iloc[0, :],
                    link="logit")  # TODO subst with EXPLANATION_TARGET
    shap.force_plot(explainer.expected_value[0], shap_values[0], test_df, link="logit")

    print('done')
    return Response(shap.values, status=200)


@api_view(['GET'])
def get_anchor(request, pk):  # TODO: changed self to request, check if correct or not
    # get model
    job = Job.objects.filter(pk=pk)[0]
    model = joblib.load(job.predictive_model.model_path)
    model = model[0]

    # load data
    training_df, test_df = get_encoded_logs(job)

    # get radom point in evaluation set
    EXPLANATION_TARGET = 1

    # get the actual explanation
    explainer = anchor_tabular.AnchorTabularExplainer(
        [True, False],  #dataset.class_names
        job.encoding.features, # = dataset.feature_names
        training_df.drop(['trace_id'], 1), #dataset.data
        {
            item: list(range(max(training_df[item])))
            for item in job.encoding.features
        }
    )
    explainer.fit(
        training_df,  #dataset.train
        [True, False], #dataset.labels_train
        test_df, #dataset.validation
        [True,False] #dataset.labels_validation
    )

    # show plot
    idx = 0
    np.random.seed(1)
    print('Prediction: ', explainer.class_names[MODEL[job.predictive_model.predictive_model][ModelActions.PREDICT.value](job, test_df)[0]])
    exp = explainer.explain_instance(test_df[idx], MODEL[job.predictive_model.predictive_model][ModelActions.PREDICT.value], threshold=0.95)
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
    print('done')
    return Response(dict(exp.names(), exp.precision(), exp.coverage()), status=200) #exp name , precision, coverage
