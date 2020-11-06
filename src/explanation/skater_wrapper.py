import os

from dtreeviz.trees import dtreeviz
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from sklearn.externals import joblib

from src.explanation.models import Explanation
from src.utils.file_service import create_unique_name


def explain(skater_exp: Explanation, training_df, test_df, explanation_target, prefix_target):
    job = skater_exp.job
    model = joblib.load(job.predictive_model.model_path)
    model = model[0]

    features = list(training_df.drop(['trace_id', 'label'], 1).columns.values)
    interpreter = Interpretation(training_df, feature_names=features)
    X_train = training_df.drop(['trace_id', 'label'], 1)
    Y_train = training_df['label'].values

    model_inst = InMemoryModel(model.predict, examples=X_train, model_type=model._estimator_type, unique_values=[1, 2],
                               feature_names=features, target_names=['label'])
    surrogate_explainer = interpreter.tree_surrogate(model_inst, seed=5)

    surrogate_explainer.fit(X_train, Y_train, use_oracle=True, prune='post', scorer_type='default')
    surrogate_explainer.class_names = features

    viz = dtreeviz(surrogate_explainer.estimator_,
                   X_train,
                   Y_train,
                   target_name='label',
                   feature_names=features,
                   orientation="TD",
                   class_names=list(surrogate_explainer.class_names),
                   fancy=True,
                   X=None,
                   label_fontsize=12,
                   ticks_fontsize=8,
                   fontname="Arial")
    name = create_unique_name("skater_plot.svg");
    viz.save(name)
    if os.path.getsize(name) > 15000000:
        return 'The file size is too big';
    f = open(name, "r")
    response = f.read()
    os.remove(name)
    if os.path.isfile(name.split('.svg')[0]):
        os.remove(name.split('.svg')[0])

    return response
