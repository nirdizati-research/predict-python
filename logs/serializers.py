from rest_framework import serializers

from logs.models import Split
from .models import Log


class LogSerializer(serializers.ModelSerializer):
    class Meta:
        model = Log
        fields = ('id', 'name')


class CreateSplitSerializer(serializers.ModelSerializer):
    config = serializers.JSONField(required=False)

    class Meta:
        model = Split
        fields = ('id', 'config', 'original_log')


class SplitSerializer(serializers.ModelSerializer):
    original_log = LogSerializer(many=False, read_only=False)
    config = serializers.JSONField()

    class Meta:
        model = Split
        fields = ('id', 'config', 'original_log', 'type')
