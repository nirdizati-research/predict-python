from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.LogList.as_view()),
    url(r'^(?P<pk>[0-9]+)/(?P<stat>events|resources|executions|traceAttributes|eventsInTrace)$', views.get_log_stats),
]

split_url_patterns = [
    url(r'^$', views.SplitList.as_view()),
    url(r'^(?P<pk>[0-9]+)$', views.SplitDetail.as_view()),
    url(r'multiple$', views.upload_multiple),
]
