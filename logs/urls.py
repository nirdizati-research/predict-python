from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.LogList.as_view()),
    url(r'^(?P<pk>[0-9]+)/(?P<stat>events|resources|executions)$', views.get_log_stats),
]
