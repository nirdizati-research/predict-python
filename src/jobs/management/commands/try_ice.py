from django.core.management.base import BaseCommand
from pdpbox.utils import _get_grids
from sklearn.externals import joblib
from pdpbox import pdp, get_dataset, info_plots
import matplotlib.pyplot as plt
import numpy as numpy
from src.core.core import get_encoded_logs
from src.encoding.common import retrieve_proper_encoder
from src.jobs.models import Job


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        # get model
        TARGET_MODEL = 5
        job = Job.objects.filter(pk=TARGET_MODEL)[0]
        model = joblib.load(job.predictive_model.model_path)[0]
        # load data
        training_df, test_df = get_encoded_logs(job)
        columns = list(training_df.columns.values)

        features = list(training_df.drop(
            ['trace_id', 'label'],
            1).columns.values)
        feature = 'prefix_1'
        feature_grids, percentile_info = _get_grids(
            feature_values=training_df[feature].values, num_grid_points=10, grid_type=None,
            percentile_range='percentile', grid_range=None)
        custom_grids = []
        indexs = []
        for x in range(int(feature_grids.min()), int(feature_grids.max() - 1)):
            custom_grids.append(x);

        fig, axes, summary_df = info_plots.target_plot(
            df=training_df,
            feature=feature,
            feature_name='prefix_2_name',
            cust_grid_points=custom_grids,
            target='label', show_percentile=False
        )
        print(summary_df)
        fig.savefig('target_plot.png')

        lists = list(training_df[feature].values)
        for x in range(int(feature_grids.min()), int(feature_grids.max() - 1)):
            indexs.append(lists.index(x))
        encoder = retrieve_proper_encoder(job)
        encoder.decode(training_df, job.encoding)
        values = training_df[feature].values
        lst = []
        summary_df
        for x in range(len(indexs) - 1):
            lst.append({'value': values[indexs[x]],
                        'label': summary_df.label.values[x],
                        'count': summary_df.values[x][4],
                        })
        print(lst)

    # explanation_target_df
    # # print(test_df)
    # print(explanation_target_df.tail(1)['prefix_1'][0])

    # pdp_fare = pdp.pdp_isolate(
    #     model=model, dataset=training_df, model_features=features, feature='prefix_1'
    # )
    # fig, axes = pdp.pdp_plot(pdp_fare, 'prefix_1')
    # fig.savefig('label.png')
    #
    # inter1 = pdp.pdp_interact(
    #     model=model, dataset=training_df, model_features=features, features=['prefix_1', 'prefix_2']
    # )
    # fig, axes = pdp.pdp_interact_plot(inter1, ['prefix_1', 'prefix_2'], plot_type='grid', x_quantile=True,
    #                                   plot_pdp=False)
    # fig.savefig('pdp_interact_plot.png')
    #
    # fig, axes, summary_df = info_plots.actual_plot_interact(
    #     model=model, X=training_df[features], features=['prefix_6', 'prefix_1'],
    #     feature_names=['prefix_6', 'prefix_1']
    # )
    # fig.savefig('actual_plot_interact.png')

