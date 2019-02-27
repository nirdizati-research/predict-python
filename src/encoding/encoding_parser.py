import re
from enum import Enum
from typing import Dict, Callable

import numpy as np
from keras.utils import to_categorical
from numpy import ndarray
from pandas import DataFrame


class Tasks(Enum):
    """
    defines the type of tasks that can be used during dataset parsing (for example when deciding if to remove some
    columns)
    """
    CLASSIFICATION = 0
    REGRESSION = 1
    TIME_SERIES_PREDICTION = 2


class EncodingParser:
    """
    parses the encoded datasets into a suitable format for the keras models (0-1 float range, one-hot encodable classes
    etc.), plus minor utils
    """

    def __init__(self, encoding: str, binary_target: bool, task: Tasks):
        """Initializes the EncodingParser

        :param encoding: encoding type
        :param binary_target: if the target is True/False or categorical
        :param task: the task type

        """
        self._encoding = encoding
        self._task = task
        self._binary_target = binary_target
        self.n_classes_x = 0
        self.n_events = 0
        self.n_event_features = 1
        self.n_classes_y = 0
        self._x_min, self._x_max, self._y_min, self._y_max = -1, -1, -1, -1

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

    def parse_targets(self, y: ndarray) -> ndarray:
        """parses the target dataset

        encodes the target dataset based on the encoding given in the init method. Stores min and max value/classes
        number based on the encoding
        :param y: input dataset
        :return: parsed input dataset

        """
        if self._task == Tasks.REGRESSION:
            self._y_min, self._y_max = np.min(y), np.max(y)
            if self._y_min == self._y_max and self._y_min == 0:
                self._y_max = 1
            elif self._y_min == self._y_max and self._y_min != 0:
                self._y_min = 0
            y = self._normalize(y, self._y_min, self._y_max)
        elif self._binary_target:
            y = y.astype(int)
        else:
            self.n_classes_y = np.max(y)
            if 0 not in y:
                self.n_classes_y += 1
            y = to_categorical(y, self.n_classes_y + 1)
        return y

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
        train_data = train_data.values
        self.n_classes_x = np.max(train_data) + 1

        if self._task == Tasks.TIME_SERIES_PREDICTION:
            train_data = np.expand_dims(train_data, -1)

        if train_data.shape[-1] == 1:
            train_data = np.expand_dims(train_data, -1)
        train_data = to_categorical(train_data, self.n_classes_x + 1)
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
        train_data = train_data.values

        self.n_classes_x = np.max(train_data) + 1
        if 0 not in train_data:
            self.n_classes_x += 1

        if self._task == Tasks.TIME_SERIES_PREDICTION:
            train_data = np.reshape(train_data, (train_data.shape[0], self.n_events, -1))
            self.n_event_features = train_data.shape[-1]

        train_data = to_categorical(train_data, self.n_classes_x + 1)
        return train_data

    def _parse_train_data_boolean(self, train_data: DataFrame) -> ndarray:
        """parses the train dataset encoded with boolean encoding

        casts each boolean to int, and stores the number of classes as the width of the dataset

        :param train_data: input dataset
        :return: parsed dataset

        """
        train_data = train_data.values
        train_data = train_data.astype(int)
        self.n_classes_x = train_data.shape[1]
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
        test_data = test_data.values
        test_data = np.clip(test_data, 0, self.n_classes_x + 1)

        if self._task == Tasks.TIME_SERIES_PREDICTION:
            test_data = np.expand_dims(test_data, -1)

        if test_data.shape[-1] == 1:
            test_data = np.expand_dims(test_data, -1)
        test_data = to_categorical(test_data, self.n_classes_x + 1)
        return test_data

    def _parse_test_data_complex(self, test_data: DataFrame) -> ndarray:
        """parses the test dataset encoded with complex encoding

        same steps as simple_encoding parsing, but with the addition of reshaping and column dropping in the time
        series prediction case

        :param test_data: input dataset
        :return: parsed dataset

        """
        test_data = self._remove_trace_attributes(test_data)

        test_data = test_data.values
        test_data = test_data.astype(int)  # TODO: fix this issue
        test_data = np.clip(test_data, 0, self.n_classes_x + 1)

        if self._task == Tasks.TIME_SERIES_PREDICTION:
            test_data = np.reshape(test_data, (test_data.shape[0], self.n_events, -1))

        test_data = to_categorical(test_data, self.n_classes_x + 1)
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
        return (predictions * (self._y_max - self._y_min)) + self._y_min

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

    @staticmethod
    def _extract_n_events(data: DataFrame) -> int:
        """returns the number of events in a trace

        simple regular expression returns the number in the last column header, indicating the last event number

        :param data: input dataframe
        :return: number of events

        """
        headers = data.columns.tolist()
        return int(re.findall('\d+', headers[-1])[0])
