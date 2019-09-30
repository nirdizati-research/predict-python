from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^prediction/', views.get_prediction),
    url(r'^demo/(?P<pk>[0-9]+)&(?P<pk1>[0-9]+)&(?P<pk2>[0-9]+)$', views.get_demo),
    url(r'^models/', views.modelList),
    url(r'^traces/', views.tracesList),
]
