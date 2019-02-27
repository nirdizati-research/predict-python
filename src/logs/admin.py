from django.contrib import admin

from src.split.models import Split
from .models import Log

admin.site.register(Log)
admin.site.register(Split)
