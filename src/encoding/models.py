from enum import Enum

from django.db import models
from django.contrib.postgres.fields import JSONField

from src.clustering.models import Clustering
from src.common.models import CommonModel
from src.labelling.models import Labelling
from src.predictive_model.models import PredictiveModel
from src.split.models import Split


class DataEncodings(Enum):
    LABEL_ENCODER = 'label_encoder'
    ONE_HOT_ENCODER = 'one_hot'


class ValueEncodings(Enum):
    SIMPLE_INDEX = 'simpleIndex'
    BOOLEAN = 'boolean'
    FREQUENCY = 'frequency'
    COMPLEX = 'complex'
    LAST_PAYLOAD = 'lastPayload'
    # SEQUENCES = 'sequences' #TODO JONAS
    DEVIANCE = 'deviance' #TODO JONAS


class TaskGenerationTypes(Enum):
    UP_TO = 'up_to'
    ONLY_THIS = 'only'
    ALL_IN_ONE = 'all_in_one'


DATA_ENCODING_MAPPINGS = (
    (DataEncodings.LABEL_ENCODER.value, 'label_encoder'),
    (DataEncodings.ONE_HOT_ENCODER.value, 'one_hot')
)

VALUE_ENCODING_MAPPINGS = (
    (ValueEncodings.SIMPLE_INDEX.value, 'simpleIndex'),
    (ValueEncodings.BOOLEAN.value, 'boolean'),
    (ValueEncodings.FREQUENCY.value, 'frequency'),
    (ValueEncodings.COMPLEX.value, 'complex'),
    (ValueEncodings.LAST_PAYLOAD.value, 'lastPayload'),
    # (ValueEncodings.SEQUENCES.value, 'sequences'), #TODO JONAS
    (ValueEncodings.DEVIANCE.value, 'deviance') #TODO JONAS
)

TASK_GENERATION_TYPE_MAPPINGS = (
    (TaskGenerationTypes.UP_TO.value, 'up_to'),
    (TaskGenerationTypes.ONLY_THIS.value, 'only_this'),
    (TaskGenerationTypes.ALL_IN_ONE.value, 'all_in_one')
)


class Encoding(CommonModel):
    data_encoding = models.CharField(choices=DATA_ENCODING_MAPPINGS, default='label_encoder', max_length=max(len(el[1]) for el in DATA_ENCODING_MAPPINGS)+1)
    value_encoding = models.CharField(choices=VALUE_ENCODING_MAPPINGS, default='simpleIndex', max_length=max(len(el[1]) for el in VALUE_ENCODING_MAPPINGS)+1)
    add_elapsed_time = models.BooleanField(default=False)
    add_remaining_time = models.BooleanField(default=False)
    add_executed_events = models.BooleanField(default=False)
    add_resources_used = models.BooleanField(default=False)
    add_new_traces = models.BooleanField(default=False)
    features = JSONField(default=dict)
    prefix_length = models.PositiveIntegerField()
    padding = models.BooleanField(default=False)
    task_generation_type = models.CharField(choices=TASK_GENERATION_TYPE_MAPPINGS, default='only_this', max_length=max(len(el[1]) for el in TASK_GENERATION_TYPE_MAPPINGS)+1)

    def to_dict(self) -> dict:
        return {
            'data_encoding': self.data_encoding,
            'value_encoding': self.value_encoding,
            'add_elapsed_time': self.add_elapsed_time,
            'add_remaining_time': self.add_remaining_time,
            'add_executed_events': self.add_executed_events,
            'add_resources_used': self.add_resources_used,
            'add_new_traces': self.add_new_traces,
            'features': self.features,
            'prefix_length': self.prefix_length,
            'padding': self.padding,
            'task_generation_type': self.task_generation_type
        }
