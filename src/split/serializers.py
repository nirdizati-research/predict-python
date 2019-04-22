from rest_framework import serializers

from src.split.models import Split, SplitTypes


class CreateSplitSerializer(serializers.ModelSerializer):
    config = serializers.JSONField(required=False)
    type = SplitTypes.SPLIT_SINGLE.value if 'type' not in config else SplitTypes.SPLIT_DOUBLE.value
    original_log = config.get('original_log', None)
    test_size = config.get('test_size', None)
    splitting_method = config.get('splitting_method', None)
    train_log = config.get('train_log', None) #TODO CREATE LOG
    test_log = config.get('test_log', None) #TODO CREATE LOG
    additional_columns = None

    class Meta:
        model = Split
        fields = ('type', 'original_log', 'test_size', 'splitting_method', 'train_log', 'test_log', 'additional_columns')


class SplitSerializer(serializers.ModelSerializer):
    training_log = serializers.SerializerMethodField()

    def get_training_log(self, split):
        return split.train_log.pk if split.train_log is not None else None

    class Meta:
        model = Split
        fields = ('id', 'original_log', 'type', 'test_log', 'training_log')
