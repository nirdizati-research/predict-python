from rest_framework import serializers

from .models import Job


class JobSerializer(serializers.ModelSerializer):
    config = serializers.JSONField()
    result = serializers.JSONField(required=False)

    class Meta:
        model = Job
        fields = ('id', 'created_date', 'modified_date', 'config', 'status', 'result')
