from collections import namedtuple

# Encoding methods
from sklearn.preprocessing import LabelEncoder

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

encoder = {}
label_dict = {}



class EncodingContainer(namedtuple('EncodingContainer', ["method", "prefix_length", "padding", "generation_type"])):
    """Inner object describing encoding configuration.
    """

    def __new__(cls, method=SIMPLE_INDEX, prefix_length=1, padding=NO_PADDING,
                generation_type=ONLY_THIS):
        return super(EncodingContainer, cls).__new__(cls, method, prefix_length, padding, generation_type)

    def is_zero_padding(self):
        return self.padding == ZERO_PADDING

    def is_all_in_one(self):
        return self.generation_type == ALL_IN_ONE

    def is_boolean(self):
        return self.method == BOOLEAN

    def is_complex(self):
        return self.method == COMPLEX

    def encode(self, df):
        for column in df:
            if column in encoder and column != 'trace_id' and column != 'label':
                #TODO workaround to avoid ValueError("y contains new labels: %s" % str(diff))
                # see https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
                df[column] = df[column].apply(lambda x: label_dict[column].get(x, label_dict[column]['MAX_VAL']))

    def init_label_encoder(self, df):
        for column in df:
            encoder[column] = LabelEncoder().fit(df[column])
            #TODO workaround to avoid ValueError("y contains new labels: %s" % str(diff))
            # see https://stackoverflow.com/questions/21057621/sklearn-labelencoder-with-never-seen-before-values
            classes = encoder[column].classes_
            transforms = encoder[column].transform(encoder[column].classes_)
            label_dict[column] = dict(zip(classes, transforms))
            label_dict[column]['MAX_VAL'] = max(transforms)+100
