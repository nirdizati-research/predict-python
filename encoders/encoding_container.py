from collections import namedtuple

# Encoding methods
SIMPLE_INDEX = 'simpleIndex'
BOOLEAN = 'boolean'
FREQUENCY = 'frequency'
COMPLEX = 'complex'
LAST_PAYLOAD = 'lastPayload'

# Generation types
UP_TO = 'up_to'
ONLY_THIS = 'only'
ALL_IN_ONE = 'all_in_one'

# padding
ZERO_PADDING = 'zero_padding'
NO_PADDING = 'no_padding'


class EncodingContainer(namedtuple('EncodingContainer', ["method", "prefix_length", "padding", "generation_type"])):
    """Inner object describing encoding configuration.
    """

    def __new__(cls, method=SIMPLE_INDEX, prefix_length=1, padding=NO_PADDING,
                generation_type=ONLY_THIS):  # TODO: fix incompatible signatures of __new__ and __init__
        return super(EncodingContainer, cls).__new__(cls, method, prefix_length, padding, generation_type)

    def is_zero_padding(self):
        return self.padding == ZERO_PADDING

    def is_all_in_one(self):
        return self.generation_type == ALL_IN_ONE

    def is_boolean(self):
        return self.method == BOOLEAN

    def is_complex(self):
        return self.method == COMPLEX
