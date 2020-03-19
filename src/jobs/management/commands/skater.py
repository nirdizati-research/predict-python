from django.core.management.base import BaseCommand
from skater.core.visualizer import plot_tree, tree_to_text
from skater.core.visualizer.tree_visualizer import _generate_graph
from sklearn.externals import joblib

from src.core.core import get_encoded_logs
from src.jobs.models import Job
import warnings
import os

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from skater.core.global_interpretation.tree_surrogate import TreeSurrogate
from sklearn import preprocessing, exceptions
from dtreeviz.trees import *
from sklearn import tree


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def dot_to_json(file_in):
        import networkx
        from networkx.readwrite import json_graph
        import pydot
        graph_netx = networkx.drawing.nx_pydot.read_dot("ss.raw")
        graph_json = json_graph.node_link_data(graph_netx)
        return json_graph.node_link_data(graph_netx)

    def handle(self, *args, **kwargs):
        # get model
        TARGET_MODEL = 52
        job = Job.objects.filter(pk=TARGET_MODEL)[0]
        model = joblib.load(job.predictive_model.model_path)[0]
        # load data
        training_df, test_df = get_encoded_logs(job)

        features = list(training_df.drop(['trace_id', 'label'], 1).columns.values)
        interpreter = Interpretation(training_df, feature_names=features)
        X_train = training_df.drop(['trace_id', 'label'], 1)
        Y_train = training_df['label'].values

        model_inst = InMemoryModel(model.predict, examples=X_train, model_type='classifier', unique_values=[1, 2],
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
        viz.save("skater_plot_train_2_2.svg")

        # graph_inst = plot_tree(surrogate_explainer._TreeSurrogate__model, 'classifier', feature_names=features,
        #                        color_list=['coral', 'lightsteelblue', 'darkkhaki'],
        #                        class_names=surrogate_explainer.class_names, enable_node_id=True, seed=0)
        # ss = surrogate_explainer.decisions_as_txt(scope='local',
        #                                           X=test_df.drop(['trace_id', 'label'], 1).iloc[12])
        # graph = _generate_graph(surrogate_explainer._TreeSurrogate__model, 'classifier', enable_node_id=True,
        #                         coverage=True)
        #
        # graph.write_raw("sss.raw")

        # graph_inst.write_raw("ss.raw")

    def viz_breast_cancer(orientation="TD",
                          max_depth=3,
                          random_state=666,
                          fancy=True,
                          pickX=False,
                          label_fontsize=12,
                          ticks_fontsize=8,
                          fontname="Arial"):
        clf = tree.DecisionTreeClassifier(
            max_depth=max_depth, random_state=random_state)
        cancer = load_breast_cancer()

        clf.fit(cancer.data, cancer.target)

        X = None
        if pickX:
            X = cancer.data[np.random.randint(0, len(cancer)), :]
        clf
        viz = dtreeviz(clf,
                       cancer.data,
                       cancer.target,
                       target_name='cancer',
                       feature_names=cancer.feature_names,
                       orientation=orientation,
                       class_names=list(cancer.target_names),
                       fancy=fancy,
                       X=X,
                       label_fontsize=label_fontsize,
                       ticks_fontsize=ticks_fontsize,
                       fontname=fontname)
        viz.save("viz.svg")
        return viz
