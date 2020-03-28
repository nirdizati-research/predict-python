from django.conf.urls import url

from src.logs.views import SplitDetail, upload_multiple, get_split_train_logs, get_split_test_logs
from src.split.views import SplitList

urlpatterns = [
    url(r'^$', SplitList.as_view()),
    url(r'^(?P<pk>[0-9]+)$', SplitDetail.as_view()),
    url(r'multiple$', upload_multiple),
    url(r'^(?P<pk>[0-9]+)/logs/train', get_split_train_logs),
    url(r'^(?P<pk>[0-9]+)/logs/test', get_split_test_logs)

]
