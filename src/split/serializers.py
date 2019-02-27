from rest_framework import serializers

from src.split.models import Split


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
