from rest_framework import serializers

from .models import ModelSplit, PredModels


class SplitSerializer(serializers.ModelSerializer):
    type = serializers.CharField()

    class Meta:
        model = ModelSplit
        fields = ('id', 'type')


class ModelSerializer(serializers.ModelSerializer):
    split = SplitSerializer(many=False, read_only=True)
    type = serializers.CharField()
    config = serializers.JSONField()

    class Meta:
        model = PredModels
        fields = ('id', 'split', 'type', 'config')
