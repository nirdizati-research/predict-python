from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.JobList.as_view()),
    url(r'^(?P<pk>[0-9]+)$', views.JobDetail.as_view()),
    url(r'multiple$', views.create_multiple),
    url(r'predict$', views.create_prediction),
]
