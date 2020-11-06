import random

from pandas import Series

from src.cache.cache import put_labelled_logs
from src.explanation.models import Explanation
from src.jobs.tasks import prediction_task
from src.logs.models import Log
from src.runtime.tasks import create_prediction_job
from src.utils.django_orm import duplicate_orm_row


def randomise_features(train_df, test_df, explanation_target):
    TARGETS = explanation_target
    for target in TARGETS:
        if len(target) == 1:
            target = target[0]
            for df in [train_df, test_df]:
                m_col = df[target[0]]
                del df[target[0]]
                target_values1 = list(set(m_col.values))
                df[target[0]] = m_col.apply(
                    lambda x:
                    x
                    if (x != target[1]) else
                    random.choice(target_values1)
                )
        elif len(target) > 1:
            for df in [train_df, test_df]:
                m_col = df[[column for column, _ in target]]
                possible_values = {}
                for column, value in target:
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
    return train_df, test_df


def save_randomised_set(initial_split_obj):
    # todo: save new dataset in memory and create split to use it
    new_split = duplicate_orm_row(initial_split_obj)

    # TODO future bug creates shadows,
    train_log = Log.objects.get_or_create(
        name='RETRAIN' + new_split.train_log.name,
        path='cache/log_cache/' + 'RETRAIN' + new_split.train_log.name,
        properties={}
    )[0]
    test_log = Log.objects.get_or_create(
        name='RETRAIN' + new_split.test_log.name,
        path='cache/log_cache/' + 'RETRAIN' + new_split.test_log.name,
        properties={}
    )[0]

    new_split.train_log = train_log
    new_split.test_log = test_log
    new_split.additional_columns = None
    new_split.save()
    return new_split


def explain(retrain_exp: Explanation, training_df_old, test_df_old, explanation_target, prefix_target):
    initial_job_obj = retrain_exp.job
    # todo: return performances
    inital_result = dict(initial_job_obj.evaluation.classificationmetrics.to_dict())  # TODO future bug

    train_df,test_df = randomise_features(training_df_old.copy(), test_df_old.copy(), explanation_target)
    assert not train_df.equals(training_df_old)
    assert not test_df.equals(test_df_old)

    new_split = save_randomised_set(initial_job_obj.split)

    prediction_job = create_prediction_job(initial_job_obj, initial_job_obj.encoding.prefix_length)
    prediction_job.split = new_split
    prediction_job.split.save()
    prediction_job.evaluation = None
    prediction_job.save()
    # assert prediction_job.split.id != initial_job_obj.split.id

    put_labelled_logs(prediction_job, train_df, test_df)

    # todo: build model
    prediction_task(prediction_job.id, do_publish_result=False)
    prediction_job.refresh_from_db()

    # todo: return performances
    return {"Initial result": inital_result, "Retrain result": prediction_job.evaluation.classificationmetrics.to_dict()}
