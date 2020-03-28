from django.conf.urls import url

from src.logs.views import SplitDetail, upload_multiple
from src.split.views import SplitList

urlpatterns = [
    url(r'^$', SplitList.as_view()),
    url(r'^(?P<pk>[0-9]+)$', SplitDetail.as_view()),
    url(r'multiple$', upload_multiple)
]
