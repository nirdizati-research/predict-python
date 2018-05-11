from collections import namedtuple
from enum import Enum


class Encoding(Enum):
    SIMPLE_INDEX = 'simpleIndex'
    BOOLEAN = 'boolean'
    FREQUENCY = 'frequency'
    COMPLEX = 'complex'
    LAST_PAYLOAD = 'lastPayload'


class GenerationType(Enum):
    UP_TO = 'up_to'
    ONLY_THIS = 'only'
    ALL_IN_ONE = 'all_in_one'


class Padding(Enum):
    ZERO_PADDING = 'zero_padding'
    NO_PADDING = 'no_padding'


class EncodingContainer(namedtuple('EncodingContainer', ["method", "prefix_length", "padding", "generation_type"])):
    """Inner object describing encoding configuration.
    """

    def __new__(cls, method=Encoding.SIMPLE_INDEX, prefix_length=1, padding=Padding.NO_PADDING,
                generation_type=GenerationType.ONLY_THIS):
        return super(EncodingContainer, cls).__new__(cls, method, prefix_length, padding, generation_type)

    def is_zero_padding(self):
        return self.padding == Padding.ZERO_PADDING

    def is_all_in_one(self):
        return self.generation_type == GenerationType.ALL_IN_ONE
