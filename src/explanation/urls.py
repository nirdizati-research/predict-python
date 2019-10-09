from django.conf.urls import url

from . import common

urlpatterns = [
    url(r'^lime/(?P<pk>[0-9]+)', common.get_lime),
    url(r'^shap/(?P<pk>[0-9]+)', common.get_shap),
    url(r'^anchor/(?P<pk>[0-9]+)', common.get_anchor),
]
