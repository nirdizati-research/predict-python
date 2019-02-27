from rest_framework import serializers

from .models import Log


class LogSerializer(serializers.ModelSerializer):
    properties = serializers.JSONField()

    class Meta:
        model = Log
        fields = ('id', 'name', 'properties')
