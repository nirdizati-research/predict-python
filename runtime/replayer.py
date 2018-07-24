import threading
from random import randint
from time import sleep
import django_rq
from logs.file_service import get_logs
from logs.models import Log
from runtime.models import DemoReplayer
from .replay_core import prepare
from rq_scheduler import Scheduler
from training.settings import RQ_QUEUES
from datetime import timedelta


scheduler = Scheduler(connection=django_rq.get_connection('default'), interval=5)


class Replayer():

    def __init__(self, id, reg_id, class_id):
        self.log_id = id
        self.class_id = class_id
        self.reg_id = reg_id
        self.log = None

    def start(self):
        replayer = DemoReplayer.objects.create(running=False)
        if replayer.running:
            return print("Replayer is already running")
        else:
            try:
                self.log = Log.objects.get(pk=self.log_id)
            except Log.DoesNotExist:
                return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)
            replayer.running = True
            replayer.save()
            self.execute(replayer.id)

    def execute(self, id):
        xlog = get_logs(self.log.path)
        # t=threading.Thread(target=self.events_list, args=(xlog, id))
        events = self.events_list(xlog, id)

    def send_events(self, trace, log, replayer):
        c = 0
        for event in trace:
            if replayer.running:
                c = c + (randint(1, 3)*5)
                scheduler.enqueue_in(timedelta(seconds=c), prepare, event, trace, log, replayer.id, self.reg_id, self.class_id, self.log)
            else:
                return
        scheduler.enqueue_in(timedelta(seconds=c), prepare, event, trace, log, replayer.id, self.reg_id, self.class_id, self.log, end=True)
        return

    def events_list(self, logs, id):
        for log in logs:
            for trace in log:
                replayer = DemoReplayer.objects.get(pk=id)
                if replayer.running:
                    t=threading.Thread(target=self.send_events, args=(trace, log, replayer))
                    t.daemon = True
                    t.start()
                else:
                    replayer.delete()
                    return
        print("Finito")
        q = django_rq.get_failed_queue()
        q.empty()
        return