from rest_framework import serializers

from logs.models import Split
from .models import Log


class LogSerializer(serializers.ModelSerializer):
    properties = serializers.JSONField()

    class Meta:
        model = Log
        fields = ('id', 'name', 'properties')


class CreateSplitSerializer(serializers.ModelSerializer):
    config = serializers.JSONField(required=False)

    class Meta:
        model = Split
        fields = ('id', 'config', 'original_log')


class SplitSerializer(serializers.ModelSerializer):
    config = serializers.JSONField()

    class Meta:
        model = Split
        fields = ('id', 'config', 'original_log', 'type', 'test_log', 'training_log')
