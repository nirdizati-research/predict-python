from django.conf.urls import url, include
from django.contrib import admin

from src.split.urls import urlpatterns

urlpatterns = [
    url(r'^jobs/', include('src.jobs.urls')),
    url(r'^logs/', include('src.logs.urls')),
    url(r'^splits/', include(urlpatterns)),
    url(r'^admin/', admin.site.urls),
    url(r'^runtime/', include('src.runtime.urls')),
    url(r'^django-rq/', include('django_rq.urls')),
]
