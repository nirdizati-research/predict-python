from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^prediction/', views.post_prediction),
    url(r'^replay/', views.post_replay),
    url(r'^replay_prediction/', views.post_replay_prediction),
    url(r'^predictions/(?P<pk>[0-9]+)&(?P<explanation_target>[0-9]+)', views.get_prediction),

]
