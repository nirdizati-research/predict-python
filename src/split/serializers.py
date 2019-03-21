from rest_framework import serializers

from src.split.models import Split


class CreateSplitSerializer(serializers.ModelSerializer):
    class Meta:
        model = Split
        fields = ('original_log', 'splitting_method', 'test_size')


class SplitSerializer(serializers.ModelSerializer):
    training_log = serializers.SerializerMethodField()

    def get_training_log(self, split):
        return split.train_log.pk if split.train_log is not None else None

    class Meta:
        model = Split
        fields = ('id', 'original_log', 'type', 'splitting_method', 'test_log', 'training_log', 'test_size')
