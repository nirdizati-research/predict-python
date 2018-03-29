from rest_framework import serializers

from logs.serializers import SplitSerializer, LogSerializer
from .models import Job


class JobSerializer(serializers.ModelSerializer):
    config = serializers.JSONField()
    result = serializers.JSONField(required=False)
    split = SplitSerializer(many=False, read_only=True)

    class Meta:
        model = Job
        fields = ('id', 'created_date', 'modified_date', 'config', 'status', 'result', 'type', 'split', 'error')