from logs.file_service import get_logs
from logs.log_service import events_by_date
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier
from django_rq import enqueue
from logs.models import Log
from time import sleep
import threading
from random import randint
from .core import prepare
from .models import Event, Trace
import django_rq

class Replayer():
    def __init__(self,id):
        self.running=False
        self.log_id=id
        self.log=None
    
    def start(self):    
        if self.running:
            return print("Replayer is already running")
        else:
            try:
                self.log=Log.objects.get(pk=self.log_id)
            except Log.DoesNotExist:
                return Response({'error': 'not in database'}, status=status.HTTP_404_NOT_FOUND)
            self.running=True
            self.execute()
    
    def execute(self):
        xlog=get_logs(self.log.path)
        events=self.events_list(xlog)
        
    def thread_fun(self, trace, log):
       for event in trace:
                django_rq.enqueue(prepare, event, trace, log)
                sleep(randint(0,1))  
                       
    def events_list(self, logs):
        for log in logs:
            for trace in log:
                t=threading.Thread(target=self.thread_fun, args=(trace,log))
                t.daemon = True 
                t.start()
        return 1          
                
    def pause(self):
        if self.running==False:
            print("The replayer is already paused")
        else:
            self.running=False
            print("the replayer is in pause")