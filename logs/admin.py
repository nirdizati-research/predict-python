from django.contrib import admin

from logs.models import Split
from .models import Log

admin.site.register(Log)
admin.site.register(Split)
