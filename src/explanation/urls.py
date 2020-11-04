from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^temporal_stability/(?P<pk>[0-9]+)&(?P<explanation_target>[0-9_]+)', views.get_temporal_stability),
    url(r'^temporal_stability/(?P<pk>[0-9]+)', views.get_temporal_stability),
    url(r'^lime_temporal_stability/(?P<pk>[0-9]+)&(?P<explanation_target>[0-9_]+)', views.get_lime_temporal_stability),
    url(r'^lime_temporal_stability/(?P<pk>[0-9_]+)', views.get_lime_temporal_stability),
    url(r'^lime/(?P<pk>[0-9]+)&(?P<explanation_target>[0-9_]+)', views.get_lime),
    url(r'^shap/(?P<pk>[0-9]+)&(?P<explanation_target>[0-9_]+)&(?P<prefix_target>.+)', views.get_shap),
    url(r'^skater/(?P<pk>[0-9]+)', views.get_skater),
    url(r'^ice/(?P<pk>[0-9]+)&(?P<explanation_target>[^/]+)', views.get_ice),
    url(r'^cmfeedback/(?P<pk>[0-9]+)&(?P<top_k>[0-9]+)', views.get_cmfeedback),
    url(r'^retrain/(?P<pk>[0-9]+)', views.get_retrain),

    # url(r'^anchor/(?P<pk>[0-9]+)', views.get_anchor), #todo not ready
]
