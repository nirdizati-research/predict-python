from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.log_list, name='logs'),
    url(r'^upload$', views.upload_file, name='upload'),
    url(r'^(?P<pk>[0-9]+)/(?P<stat>events|resources|executions)$', views.get_log_stats),
]
