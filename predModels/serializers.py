from rest_framework import serializers
from logs.serializers import LogSerializer
from .models import ModelSplit, PredModels


class SplitSerializer(serializers.ModelSerializer):
    model_path = serializers.CharField()
    estimator_path = serializers.CharField()
    type = serializers.CharField()

    class Meta:
        model = ModelSplit
        fields = ('id', 'type', 'model_path', 'estimator_path')
        
class ModelSerializer(serializers.ModelSerializer):
    split = SplitSerializer(many=False, read_only=True)
    type = serializers.CharField()
    log = LogSerializer(many=False, read_only=True)
    config = serializers.JSONField()

    class Meta:
        model = PredModels
        fields = ('id', 'split', 'type', 'log', 'config')