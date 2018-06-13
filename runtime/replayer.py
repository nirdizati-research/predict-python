import threading
from random import randint
from time import sleep
import django_rq
from logs.file_service import get_logs
from logs.models import Log
from runtime.models import DemoReplayer
from .replay_core import prepare


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

    def thread_fun(self, trace, log, replayer):
        for event in trace:
            if replayer.running:
                django_rq.enqueue(prepare, event, trace, log, replayer.id, self.reg_id, self.class_id, self.log)
                sleep(randint(5, 20))
            else:
                return
        django_rq.enqueue(prepare, event, trace, log, replayer.id, self.reg_id, self.class_id, self.log, end=True)
        return

    def events_list(self, logs, id):
        for log in logs:
            for trace in log:
                replayer = DemoReplayer.objects.get(pk=id)
                if replayer.running:
                    t=threading.Thread(target=self.thread_fun, args=(trace,log, replayer))
                    t.daemon = True
                    t.start()
                else:
                    replayer.delete()
                    return
        return