import random

from django.core.management.base import BaseCommand
from pandas import Series

from src.cache.cache import put_labelled_logs
from src.core.core import get_encoded_logs

from src.jobs.models import Job
from src.jobs.tasks import prediction_task
from src.runtime.tasks import create_prediction_job
from src.utils.django_orm import duplicate_orm_row


class Command(BaseCommand):
    help = 'tries to deliver an explanation of a random prediction of the trained model'

    def handle(self, *args, **kwargs):
        TARGET_JOB = 439
        initial_job_obj = Job.objects.filter(pk=TARGET_JOB)[0]

        # todo: return performances
        print('Initial Job:', initial_job_obj.evaluation.classificationmetrics)  # TODO future bug

        training_df_old, test_df_old = get_encoded_logs(initial_job_obj)
        training_df = training_df_old.copy()
        test_df = test_df_old.copy()

        # todo: what should I randomise?
        TARGETS = [
            [('prefix_1', 2)], # <- simple pattern
            [('prefix_2', 3)], # <- simple pattern
            [('prefix_3', 2),
            ('prefix_4', 3),] # <- complex pattern
        ]
        for target in TARGETS:
            if len(target) == 1:
                target = target[0]
                for df in [training_df, test_df]:
                    m_col = df[target[0]]
                    del df[target[0]]
                    target_values1 = list(set(m_col.values))
                    df[target[0]] = m_col.apply(
                        lambda x:
                            x if (x != target[1])
                            else random.choice(target_values1)
                    )
            elif len(target) > 1:
                for df in [training_df, test_df]:
                    m_col = df[[column for column, _ in target]]
                    possible_values = {}
                    for column, _ in target:
                        possible_values[column] = list(set(df[column]))
                        del df[column]
                    df[[column for column, _ in target]] = m_col.apply(
                        lambda x:
                            x if any([x[column] != value for column, value in target])
                            else Series({
                                column: random.choice(possible_values[column])
                                for column, value in target
                            }),
                        axis=1)
            else:
                raise Exception('target list with unexpected value')

        assert not training_df.equals(training_df_old)
        assert not test_df.equals(test_df_old)

        # todo: save new dataset in memory and create split to use it
        initial_split_obj = initial_job_obj.split
        new_split = duplicate_orm_row(initial_split_obj)
        train_log = duplicate_orm_row(new_split.train_log)
        test_log = duplicate_orm_row(new_split.test_log)

        # TODO future bug creates shadows
        train_log.name = 'RETRAIN' + train_log.name
        train_log.path = 'cache/log_cache/' + train_log.name
        train_log.properties = {}
        test_log.name = 'RETRAIN' + test_log.name
        test_log.path = 'cache/log_cache/' + test_log.name
        test_log.properties = {}

        new_split.train_log = train_log
        new_split.test_log = test_log
        new_split.additional_columns = None
        new_split.save()

        prediction_job = create_prediction_job(initial_job_obj, initial_job_obj.encoding.prefix_length)
        prediction_job.split = new_split
        prediction_job.split.save()
        prediction_job.save()

        put_labelled_logs(prediction_job, training_df, test_df)

        # todo: build model
        prediction_task(prediction_job.id, do_publish_result=False)
        prediction_job.refresh_from_db()

        # todo: return performances
        print('Retrain Job:', prediction_job.evaluation.classificationmetrics)


        print('Done, cheers!')

