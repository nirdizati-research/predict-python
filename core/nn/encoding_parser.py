from types import FunctionType
from typing import Dict

import numpy as np
from keras.utils import to_categorical
from numpy import ndarray
from pandas import DataFrame


class EncodingParser:
    def __init__(self, encoding: str, binary_target: bool, regression_task: bool):
        self._encoding = encoding
        self._regression_task = regression_task
        self._binary_target = binary_target
        self.n_classes_x = 0
        self.n_classes_y = 0
        self._x_min, self._x_max, self._y_min, self._y_max = -1, -1, -1, -1

        self._training_parsing_functions: Dict[str, FunctionType] = {
            'simpleIndex': self._parse_train_data_simple_index,
            'boolean': self._parse_train_data_boolean,
            'frequency': self._parse_train_data_frequency,
            'complex': self._parse_train_data_simple_index,
            'lastPayload': self._parse_train_data_simple_index,
        }

        self._testing_parsing_functions: Dict[str, FunctionType] = {
            'simpleIndex': self._parse_test_data_simple_index,
            'boolean': self._parse_test_data_boolean,
            'frequency': self._parse_test_data_frequency,
            'complex': self._parse_test_data_simple_index,
            'lastPayload': self._parse_test_data_simple_index,
        }

    def parse_training_dataset(self, train_data: DataFrame) -> ndarray:
        if self._encoding in self._training_parsing_functions:
            parsing_function = self._training_parsing_functions[self._encoding]
            return parsing_function(train_data)
        else:
            raise NotImplementedError('encoding method not parsable yet.')

    def parse_y(self, y: DataFrame) -> ndarray:
        y = y.values
        if self._regression_task:
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
        if self._encoding in self._testing_parsing_functions:
            parsing_function = self._testing_parsing_functions[self._encoding]
            test_data = parsing_function(test_data)
        else:
            raise NotImplementedError('encoding method not parsable yet.')

        return test_data

    def _parse_train_data_simple_index(self, train_data: DataFrame) -> ndarray:
        train_data = train_data.values
        self.n_classes_x = np.max(train_data)
        if 0 not in train_data:
            self.n_classes_x += 1
        return train_data

    def _parse_train_data_boolean(self, train_data: DataFrame) -> ndarray:
        train_data = train_data.values
        train_data = train_data.astype(int)
        self.n_classes_x = train_data.shape[1]
        return train_data

    def _parse_train_data_frequency(self, train_data: DataFrame) -> ndarray:
        train_data = train_data.values
        self._x_min, self._x_max = np.min(train_data), np.max(train_data)
        train_data = self._normalize(train_data, self._x_min, self._x_max)
        return train_data

    def _parse_test_data_simple_index(self, test_data: DataFrame) -> ndarray:
        test_data = test_data.values
        test_data = np.clip(test_data, 0, self.n_classes_x + 1)
        return test_data

    @staticmethod
    def _parse_test_data_boolean(test_data: DataFrame) -> ndarray:
        test_data = test_data.values
        test_data = test_data.astype(int)
        return test_data

    def _parse_test_data_frequency(self, test_data: DataFrame) -> ndarray:
        test_data = test_data.values
        test_data = self._normalize(test_data, self._x_min, self._x_max)
        test_data = np.clip(test_data, 0, 1)
        return test_data

    def denormalize_predictions(self, predictions: ndarray) -> ndarray:
        return self._denormalize(predictions, self._y_min, self._y_max)

    @staticmethod
    def _normalize(data: ndarray, data_min: float, data_max: float) -> ndarray:
        return (data - data_min) / (data_max - data_min)

    @staticmethod
    def _denormalize(data: ndarray, data_min: float, data_max: float) -> ndarray:
        return (data * (data_max - data_min)) + data_min
