from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^lime/(?P<pk>[0-9]+)&(?P<explanation_target>[0-9]+)', views.get_lime),
    url(r'^shap/(?P<pk>[0-9]+)', views.get_shap),
    url(r'^anchor/(?P<pk>[0-9]+)', views.get_anchor),
]
