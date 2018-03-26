from rest_framework import serializers
from logs.serializers import LogSerializer
from .models import Split, PredModels


class SplitSerializer(serializers.ModelSerializer):
    model_path = serializers.CharField()
    kmean_path = serializers.CharField()
    type = serializers.CharField()

    class Meta:
        model = Split
        fields = ('id', 'type', 'model_path', 'kmean_path')
        
class ModelSerializer(serializers.ModelSerializer):
    split = SplitSerializer(many=False, read_only=True)
    type = serializers.CharField()
    log = LogSerializer(many=False, read_only=True)
    prefix_length = serializers.IntegerField()
    encoding = serializers.CharField()
    method = serializers.CharField

    class Meta:
        model = PredModels
        fields = ('id', 'split', 'type', 'log', 'prefix_length', 'encoding', 'method')