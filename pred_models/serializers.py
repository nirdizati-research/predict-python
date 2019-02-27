from rest_framework import serializers

from src.split.serializers import SplitSerializer
from .models import PredModels


class ModelSerializer(serializers.ModelSerializer):
    split = SplitSerializer(many=False, read_only=True)
    type = serializers.CharField()
    config = serializers.JSONField()

    class Meta:
        model = PredModels
        fields = ('id', 'split', 'type', 'config')
