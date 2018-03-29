from django.conf.urls import url, include
from django.contrib import admin
from replayer.views import demo
from .views import ModelList

from logs.urls import split_url_patterns

urlpatterns = [
    url(r'^jobs/', include('jobs.urls')),
    url(r'^logs/', include('logs.urls')),
    url(r'^splits/', include(split_url_patterns)),
    url(r'^models/', ModelList),
    url(r'^demo/(?P<pk>[0-9]+)$', demo),
    url(r'^admin/', admin.site.urls),
    url(r'^django-rq/', include('django_rq.urls')),
]
