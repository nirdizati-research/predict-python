import django_rq
from django.core.management.base import BaseCommand

from src.jobs import tasks
from src.jobs.models import JobStatuses, Job


class Command(BaseCommand):
    help = 'helps requeue properly jobs that have been remove from both default and failed queue in redis'

    def handle(self, *args, **kwargs):
        errored_jobs = Job.objects.filter(status=JobStatuses.ERROR.value)
        for j in errored_jobs:
            j.status = JobStatuses.CREATED.value
            j.error = ''
            j.save()
        jobs_to_requeue = [j.id for j in errored_jobs ]
        print('Requeue of', jobs_to_requeue)
        [ django_rq.enqueue(tasks.prediction_task, j) for j in jobs_to_requeue ]
        print('done')

