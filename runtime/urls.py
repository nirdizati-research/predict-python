from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^prediction/(?P<pk1>[0-9]+)&(?P<pk2>[0-9]+)&(?P<pk3>[0-9]+)&(?P<pk4>[0-9]+)$', views.get_prediction),
    url(r'^demo/(?P<pk>[0-9]+)$', views.get_demo),
]