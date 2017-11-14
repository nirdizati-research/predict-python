from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.job_list, name='jobs'),
    url(r'^(?P<pk>[0-9]+)$', views.job_detail),
]
