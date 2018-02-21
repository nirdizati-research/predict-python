from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.JobList.as_view()),
    url(r'^(?P<pk>[0-9]+)$', views.JobDetail.as_view()),
    url(r'^training/(?P<pk>[0-9]+)$', views.get_model),
    url(r'^predict/(?P<pk1>[0-9]+)&(?P<pk2>[0-9]+)$', views.get_prediction),
    url(r'multiple$', views.create_multiple),
    url(r'predict$', views.create_prediction),
]
