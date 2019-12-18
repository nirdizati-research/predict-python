from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.LogList.as_view()),
    url(r'^(?P<pk>[0-9]+)$', views.LogDetail.as_view()),
    url(r'^(?P<pk>[0-9]+)/traces', views.get_log_traces_attributes),
    # url(r'^(?P<pk>[0-9]+)/(?P<stat>events|resources|executions|traceAttributes|eventsInTrace|newTraces)$',
    #     views.get_log_stats),
]
