from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.JobList.as_view()),
    url(r'^(?P<pk>[0-9]+)$', views.JobDetail.as_view()),
    url(r'multiple$', views.create_multiple),
    url(r'^decode/(?P<pk>[0-9]+)', views.get_decoded_df),

]
