from rest_framework import serializers
from .models import XTrace, XLog

class TraceSerializer(serializers.ModelSerializer):
    real_log = serializers.IntegerField()
    completed = serializers.BooleanField()
    first_event = serializers.DateTimeField()
    last_event = serializers.DateTimeField()
    n_events = serializers.IntegerField()
    reg_results = serializers.JSONField()
    class_results = serializers.JSONField()
    reg_actual = serializers.JSONField()
    class_actual = serializers.JSONField()
    duration = serializers.IntegerField()


    class Meta:
        model = XTrace
        fields = (
            'id', 'completed', 'real_log', 'first_event', 'last_event', 'n_events', 'reg_results', 'class_results', 'reg_actual', 'class_actual', 'duration')