from django.db import models

class Log(models.Model):
    """A XES log file on disk"""
    name = models.CharField(max_length=200)
    path = models.CharField(max_length=200)
