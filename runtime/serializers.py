from rest_framework import serializers
from logs.serializers import LogSerializer
from predModels.serializers import ModelSerializer
from .models import XTrace, XLog
        
class XLogSerializer(serializers.ModelSerializer):
    config = serializers.JSONField()

    class Meta:
        model = XLog
        fields = ('id', 'config')       
        
class TraceSerializer(serializers.ModelSerializer):
    xlog = XLogSerializer(many=False, read_only=True)
    config = serializers.JSONField()
    real_log = serializers.IntegerField()
    first_event = serializers.DateTimeField()
    last_event = serializers.DateTimeField()
    n_events = serializers.IntegerField()
    reg_model = ModelSerializer(many=False, read_only=True)
    class_model = ModelSerializer(many=False, read_only=True)
    reg_results = serializers.JSONField()
    class_results = serializers.JSONField()

    class Meta:
        model = XTrace
        fields = ('id',  'xlog', 'real_log', 'config', 'first_event', 'last_event', 'n_events', 'reg_model', 'class_model', 'reg_results', 'class_results')