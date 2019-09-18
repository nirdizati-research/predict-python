import bisect
import re
from enum import Enum
from typing import Dict, Callable

import numpy as np
from keras.utils import to_categorical
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from src.predictive_model.models import PredictiveModels


class DataEncoder:
    """
    support class for EncodingParser, tasked with actual parsing/one-hot encoding
    """

    class DataTypes(Enum):
        """
        possible data types for each column
        """
        CATEGORICAL = 'categorical'
        NUMERIC = 'numeric'

    _unknown_token = '<unknown>'

    def __init__(self, task: PredictiveModels, is_targets_dataset: bool = False):
        """initializes the DataEncoder

        :param task: task type (class, reg, time_series_pred.)
        :param is_targets_dataset: flag that indicates wether this DataEncoder is working with the targets dataset

        """
        self._data_encoders = {}
        self._task = task
        self._is_targets_dataset = is_targets_dataset
        self._base_headers = None
        self._numerical_headers = []
        self._categorical_headers = []

    def build_encoders(self, data: DataFrame) -> None:
        """builds an encoder for each column

        first the base headers are extracted (prefix_1 -> prefix, org:resources:Amount_1 -> org_resources:Amount) and
        then a dictionary of LabelEncoders is built. Numerical data stores min and max instead of a LabelEncoder.

        :param data: input dataframe

        """
        self._base_headers = self._extract_base_headers(data)

        for base_header in self._base_headers:
            relevant_data = self._get_relevant_columns(data, base_header)
            data_type = self._get_data_type(relevant_data)

            if data_type == DataEncoder.DataTypes.NUMERIC.value:
                self._numerical_headers.append(base_header)

                data_min = np.min(relevant_data.values.astype(np.float32))
                data_max = np.max(relevant_data.values.astype(np.float32))

                self._data_encoders[base_header] = {'data_type': data_type,
                                                    'label_encoder': {
                                                        'min': data_min,
                                                        'max': data_max}}
            else:
                self._categorical_headers.append(base_header)

                label_encoder = LabelEncoder()
                label_encoder.fit(relevant_data.values.flatten().tolist())
                if base_header != 'label':
                    label_encoder_classes = label_encoder.classes_.tolist()
                    bisect.insort_left(label_encoder_classes, self._unknown_token)
                    label_encoder.classes_ = label_encoder_classes
                self._data_encoders[base_header] = {'data_type': data_type,
                                                    'label_encoder': label_encoder}

    def encode_data(self, data: DataFrame, train: bool = True) -> None:
        """encodes the input data

        actual data encoding, using the built encoders. For each column type the right encoding is done
        (to class/normalization)

        :param data: input dataframe
        :param train: flag indicating whether the input is a train dataframe or a test one

        """
        for base_header in self._base_headers:
            relevant_data = self._get_relevant_columns(data, base_header)
            for column in relevant_data:
                if self._data_encoders[base_header]['data_type'] == DataEncoder.DataTypes.NUMERIC.value:
                    data_min = self._data_encoders[base_header]['label_encoder']['min']
                    data_max = self._data_encoders[base_header]['label_encoder']['max']
                    if data_min != data_max:
                        data[column] = EncodingParser._normalize(data[column].values.astype(np.float32), data_min,
                                                                 data_max)
                    else:
                        data[column] = data[column] * 0

                    data[column] = np.clip(data[column], 0.0, 1.0)
                else:
                    label_encoder = self._data_encoders[base_header]['label_encoder']

                    if not train:
                        data[column] = data[column].map(
                            lambda s: self._unknown_token if s not in label_encoder.classes_ else s)
                    data[column] = label_encoder.transform(data[column].values.tolist())

    def to_one_hot(self, data: DataFrame) -> ndarray:
        """one hot encoding

        transforms the encoded data into the one-hot representation

        :param data: input dataframe
        :return: one-hot encoded array

        """
        n_classes = self._get_highest_class_number()

        if not self._is_targets_dataset:
            n_classes += 1

        dataset = np.zeros((data.shape[0], data.shape[1], n_classes))

        for index, header in enumerate(data):
            base_header = self._extract_base_header(header)
            data_type = self._data_encoders[base_header]['data_type']
            if data_type == DataEncoder.DataTypes.CATEGORICAL.value:
                dataset[:, index, :] = to_categorical(data[header], n_classes)
            elif data_type == DataEncoder.DataTypes.NUMERIC.value:
                dataset[:, index, -1] = data[header]
        return dataset

    def _get_highest_class_number(self):
        """returns the highest class number for the used dataframe

        returns the highest class number from all the stored LabelEncoders

        :return: highest class number

        """
        n_classes_max = 0

        for header in self._categorical_headers:
            header_max = len(self._data_encoders[header]['label_encoder'].classes_)
            if header_max > n_classes_max:
                n_classes_max = header_max

        return n_classes_max

    def get_n_classes_x(self):
        """returns the number of training/test classes

        returns the highest number of classes for the encoded dataframe, adding 1 if there are numerical values.
        The structure is [one-hot encoding, normalized_value] for each variable, such that a categorical variable
        becomes [0 0 0 1 0.0] where a numerical value becomes [0 0 0 0 0 0.263]

        :return: number of training/test classes + 1 (for numerical values)

        """
        n_classes = self._get_highest_class_number()

        if not self._is_targets_dataset:
            n_classes += 1
        return n_classes

    def _get_data_type(self, data: DataFrame) -> str:
        """returns the type for the input dataframe

        tries to cast the dataframe to float, to decide wether the input contains a string or a number. Returns the
        appropriate type

        :param data: selected columns of the dataframe
        :return: data type

        """
        if len(data.columns) == 1 and 'label' in data:
            if self._task == PredictiveModels.CLASSIFICATION.value:
                return DataEncoder.DataTypes.CATEGORICAL.value

            if self._task == PredictiveModels.REGRESSION.value:
                return DataEncoder.DataTypes.NUMERIC.value

        try:
            data.values.astype(np.float32)
            return DataEncoder.DataTypes.NUMERIC.value
        except:
            return DataEncoder.DataTypes.CATEGORICAL.value

    def get_numerical_limits(self, header='label'):
        """returns the numerical limits for the input header

        returns the min and max value from the stored LabelEncoders, using header as index

        :param header: label associated with the data we want to extract min and max from
        :return: min and max values associated with the column _header_

        """
        min_value = self._data_encoders[header]['label_encoder']['min']
        max_value = self._data_encoders[header]['label_encoder']['max']
        return min_value, max_value

    @staticmethod
    def _get_relevant_columns(data: DataFrame, header: str) -> DataFrame:
        """returns the columns associated with the base header

        filters the input dataframe in order to extract all the columns matching the regex "header + (_[0-9]+)*"

        :param data: input dataframe
        :param header: base header to match
        :return: matched columns

        """
        return data.filter(regex=header + '(\_[0-9]+)?$')

    @staticmethod
    def _extract_base_headers(data: DataFrame) -> set:
        """extract the base headers

        extract the base headers from the headers of the input dataframe
        (prefix_1, prefix_2, org:resource_2 -> [prefix, org:resource])

        :param data: input dataframe
        :return: base headers

        """
        headers = data.columns.tolist()
        base_headers = set([DataEncoder._extract_base_header(header) for header in headers])
        return base_headers

    @staticmethod
    def _extract_base_header(header: str) -> str:
        """extracts the base header

        applies a regex expression to remove trailing _[0-9]* values from the input header

        :param header: header to extract base header from
        :return: base header

        """
        return re.sub(r'\_[0-9]+$', '', header)


class EncodingParser:
    """
    parses the encoded datasets into a suitable format for the keras models (0-1 float range, one-hot encodable classes
    etc.), plus minor utils
    """

    def __init__(self, encoding: str, binary_target: bool, task: PredictiveModels):
        """initializes the EncodingParser

        :param encoding: encoding type
        :param binary_target: if the target is True/False or categorical
        :param task: the task type

        """

        self._encoding = encoding
        self._task = task
        self._binary_target = binary_target
        self.n_events = 0
        self.n_event_features = 1
        self._x_min, self._x_max = -1, -1

        self._training_data_encoder = DataEncoder(self._task)
        self._target_data_encoder = DataEncoder(self._task, True)

        self._training_parsing_functions: Dict[str, Callable] = {
            'simpleIndex': self._parse_train_data_simple_index,
            'boolean': self._parse_train_data_boolean,
            'frequency': self._parse_train_data_frequency,
            'complex': self._parse_train_data_complex,
            'lastPayload': self._parse_train_data_simple_index,
        }

        self._testing_parsing_functions: Dict[str, Callable] = {
            'simpleIndex': self._parse_test_data_simple_index,
            'boolean': self._parse_test_data_boolean,
            'frequency': self._parse_test_data_frequency,
            'complex': self._parse_test_data_complex,
            'lastPayload': self._parse_test_data_simple_index,
        }

    def parse_training_dataset(self, train_data: DataFrame) -> ndarray:
        """parses the training dataset

        encodes the training dataset based on the encoding given in the init method
        :param train_data: input dataset
        :return: parsed input dataset

        """
        if self._encoding in self._training_parsing_functions:
            parsing_function = self._training_parsing_functions[self._encoding]
            parsed_dataset = parsing_function(train_data)
            return parsed_dataset
        raise NotImplementedError('encoding method not parsable yet.')

    def parse_targets(self, targets: DataFrame) -> ndarray:
        """parses the target dataset

        encodes the target dataset based on the encoding given in the init method. Stores min and max value/classes
        number based on the encoding
        :param targets: input dataset
        :return: parsed input dataset

        """

        self._target_data_encoder.build_encoders(targets)
        self._target_data_encoder.encode_data(targets)

        if self._task == PredictiveModels.CLASSIFICATION.value:
            targets = self._target_data_encoder.to_one_hot(targets)
            targets = np.squeeze(targets, 1)
        else:
            targets = targets.values
        return targets

    def parse_testing_dataset(self, test_data: DataFrame) -> ndarray:
        """parses the test dataset

        encodes the test dataset based on the encoding given in the init method
        :param test_data: input dataset
        :return: parsed input dataset

        """
        if self._encoding in self._testing_parsing_functions:
            parsing_function = self._testing_parsing_functions[self._encoding]
            return parsing_function(test_data)
        raise NotImplementedError('encoding method not parsable yet.')

    def _parse_train_data_simple_index(self, train_data: DataFrame) -> ndarray:
        """parses the train dataset encoded with simple_index encoding

        in this case it just stores the total amount of classes in the dataset

        :param train_data: input dataset
        :return: parsed dataset

        """

        self._training_data_encoder.build_encoders(train_data)
        self._training_data_encoder.encode_data(train_data)
        train_data = self._training_data_encoder.to_one_hot(train_data)

        if self._task == PredictiveModels.TIME_SERIES_PREDICTION.value:
            train_data = np.expand_dims(train_data, -2)

        return train_data

    def _parse_train_data_complex(self, train_data: DataFrame) -> ndarray:
        """parses the train dataset encoded with complex encoding

        the parsing is similar to simple_index, but for time series prediction some reshaping and column dropping is
        needed

        :param train_data: input dataset
        :return: parsed dataset

        """

        train_data = self._remove_trace_attributes(train_data)
        self.n_events = self._extract_n_events(train_data)
        self._training_data_encoder.build_encoders(train_data)
        self._training_data_encoder.encode_data(train_data)

        self.n_event_features = int(train_data.shape[-1] / self.n_events)
        train_data = self._training_data_encoder.to_one_hot(train_data)

        if self._task == PredictiveModels.TIME_SERIES_PREDICTION.value:
            train_data = np.reshape(train_data, (train_data.shape[0], self.n_events, self.n_event_features, -1))

        return train_data

    def _parse_train_data_boolean(self, train_data: DataFrame) -> ndarray:
        """parses the train dataset encoded with boolean encoding

        casts each boolean to int, and stores the number of classes as the width of the dataset

        :param train_data: input dataset
        :return: parsed dataset

        """

        train_data = train_data.values
        train_data = train_data.astype(int)
        return train_data

    def _parse_train_data_frequency(self, train_data: DataFrame) -> ndarray:
        """parses the train dataset encoded with frequency encoding

        stores min and max values, then normalizes all the values in the 0-1 range

        :param train_data: input dataset
        :return: parsed dataset

        """

        train_data = train_data.values
        self._x_min, self._x_max = np.min(train_data), np.max(train_data)
        train_data = self._normalize(train_data, self._x_min, self._x_max)
        return train_data

    def _parse_test_data_simple_index(self, test_data: DataFrame) -> ndarray:
        """parses the test dataset encoded with simple_index encoding

        clips the dataset in the 0, classes+1 range, to obtain out_of_vocabulary classes corresponding to classes + 1

        :param test_data: input dataset
        :return: parsed dataset

        """

        self._training_data_encoder.encode_data(test_data, train=False)
        test_data = self._training_data_encoder.to_one_hot(test_data)

        if self._task == PredictiveModels.TIME_SERIES_PREDICTION.value:
            test_data = np.expand_dims(test_data, -2)
        return test_data

    def _parse_test_data_complex(self, test_data: DataFrame) -> ndarray:
        """parses the test dataset encoded with complex encoding

        same steps as simple_encoding parsing, but with the addition of reshaping and column dropping in the time
        series prediction case

        :param test_data: input dataset
        :return: parsed dataset

        """
        test_data = self._remove_trace_attributes(test_data)

        self._training_data_encoder.encode_data(test_data, train=False)
        test_data = self._training_data_encoder.to_one_hot(test_data)

        if self._task == PredictiveModels.TIME_SERIES_PREDICTION.value:
            test_data = np.reshape(test_data, (test_data.shape[0], self.n_events, self.n_event_features, -1))
        return test_data

    @staticmethod
    def _parse_test_data_boolean(test_data: DataFrame) -> ndarray:
        """parses the test dataset encoded with boolean encoding

        casts each boolean to int

        :param test_data: input dataset
        :return: parsed dataset

        """
        test_data = test_data.values
        test_data = test_data.astype(int)
        return test_data

    def _parse_test_data_frequency(self, test_data: DataFrame) -> ndarray:
        """parses the test dataset encoded with frequency encoding

        normalizes the values using the stored min and max, then clips them in the 0-1 range

        :param test_data: input dataset
        :return: parsed dataset

        """
        test_data = test_data.values
        test_data = self._normalize(test_data, self._x_min, self._x_max)
        test_data = np.clip(test_data, 0, 1)
        return test_data

    def denormalize_predictions(self, predictions: ndarray) -> ndarray:
        """denormalizes the predictive_model predictions

        denormalizes the predictions using the stored y min and max

        :param predictions: predictive_model predictions
        :return: denormalized predictions

        """
        y_min, y_max = self._target_data_encoder.get_numerical_limits()
        return (predictions * (y_max - y_min)) + y_min

    @staticmethod
    def _normalize(data: ndarray, data_min: float, data_max: float) -> ndarray:
        """normalizes the input dataset

        normalizes the dataset using the given data min and max

        :type data: input dataset
        :type data_min: minimum
        :type data_max: maximum
        :return: normalized dataset

        """
        return (data - data_min) / (data_max - data_min)

    @staticmethod
    def _remove_trace_attributes(data: DataFrame) -> DataFrame:
        """removes the unnecessary traces attributes (used in the time series prediction case) using keyword matching

        :param data: input dataframe
        :return: dataframe with trace attributes removed

        """
        headers = data.columns.tolist()
        # this is required in oder to remove the first n trace attributes trace attributes
        # TODO: improve trace attribute removal
        trace_attribute_last_index = headers.index('prefix_1')
        return data.iloc[:, trace_attribute_last_index:]

    def get_n_classes_x(self):
        return self._training_data_encoder.get_n_classes_x()

    @staticmethod
    def _extract_n_events(data: DataFrame) -> int:
        """returns the number of events in a trace

        simple regular expression returns the number in the last column header, indicating the last event number

        :param data: input dataframe
        :return: number of events

        """
        headers = data.columns.tolist()
        return int(re.findall('\d+', headers[-1])[0])
