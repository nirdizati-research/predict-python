import django_rq
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from pred_models.models import PredModels
from src.jobs.models import Job, JobStatuses, JobTypes
from src.jobs.serializers import JobSerializer
from src.split.models import Split
from .models import XTrace
from .replayer import Replayer
from .serializers import TraceSerializer
from .tasks import runtime_task


@api_view(['GET'])
def tracesList(request):  # TODO: changed self to request, check if correct or not
    traces = XTrace.objects.all()
    serializer = TraceSerializer(traces, many=True)
    return Response(serializer.data, status=200)


@api_view(['GET'])
def modelList(request):
    completed_jobs = Job.objects.filter(
        status=JobStatuses.COMPLETED.value,
        type=JobTypes.PREDICTION.value,
        create_models=True
    )
    serializer = JobSerializer(completed_jobs, many=True)
    return Response(serializer.data, status=200)


@api_view(['GET'])
def get_demo(request, pk, pk1, pk2):
    replay = Replayer(pk, pk1, pk2)
    replay.start()

    return Response("Finito")


@api_view(['POST'])
def get_prediction(request):
    jobs = []
    data = request.data
    splitId = int(data['splitId'])
    regId = int(data['regId'])
    classId = int(data['classId'])
    timeSeriesPredId = int(data['timeSeriesPredId'])

    split = Split.objects.get(pk=splitId)

    try:
        if regId > 0:
            job = Job.objects.get(pk=regId)
            jobs.append(job)
        if classId > 0:
            job = Job.objects.get(pk=classId)
            jobs.append(job)
        if timeSeriesPredId > 0:
            job = Job.objects.get(pk=timeSeriesPredId)
            jobs.append(job)
    except PredModels.DoesNotExist:
        return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)

    for job in jobs:
        jobtoenqueue = generate_run(job, split)

        # django_rq.enqueue(training, jobrun, predictive_model)
        django_rq.enqueue(runtime_task, jobtoenqueue)
    # os.system('python3 manage.py rqworker --burst')
    serializer = JobSerializer(jobs, many=True)
    return Response(serializer.data, status=201)


def generate_run(job: Job, split: Split) -> Job:
    return Job.objects.get_or_create(
                    status=JobStatuses.CREATED.value,
                    type=JobTypes.RUNTIME.value,
                    encoding=job.encoding,
                    labelling=job.labelling,
                    clustering=job.clustering,
                    hyperparameter_optimizer=job.hyperparameter_optimizer,
                    predictive_model=job.predictive_model,
                    split=split,
                    create_models=False)[0]

