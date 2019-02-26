"""
neural network models (nn-classification, nn-regression and rnn-time_series_prediction)
"""

from typing import Dict, Union

import numpy as np
from keras import Input, Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import LSTM, GRU, Reshape
from numpy import ndarray
from pandas import DataFrame
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

from encoders.encoding_parser import EncodingParser, Tasks


class NNClassifier(ClassifierMixin):
    """
    Neural Network classifier, implements the same methods as the sklearn models to make it simple to add
    """

    # noinspection PyTypeChecker
    def __init__(self, **kwargs: Dict[str, Union[int, str, float]]):
        """initializes the Neural Network classifier

        :param kwargs: configuration containing the model parameters, encoding and training parameters

        """
        self._n_hidden_layers = int(kwargs['n_hidden_layers'])
        self._n_hidden_units = int(kwargs['n_hidden_units'])
        self._activation = str(kwargs['activation'])
        self._n_epochs = int(kwargs['n_epochs'])
        self._encoding = str(kwargs['encoding'])
        self._dropout_rate = float(kwargs['dropout_rate'])
        self._is_binary_classifier = bool(kwargs['is_binary_classifier'])
        self._encoding_parser = EncodingParser(self._encoding, self._is_binary_classifier, task=Tasks.CLASSIFICATION)
        self._model = None

    def fit(self, train_data: DataFrame, targets: ndarray) -> None:
        """creates and fits the model

        first the encoded data is parsed, then the model created and then trained

        :param train_data: encoded training dataset
        :param targets: encoded target dataset

        """
        train_data = self._encoding_parser.parse_training_dataset(train_data)

        targets = self._encoding_parser.parse_targets(targets)

        model_inputs = Input(train_data.shape[1:])
        predicted = model_inputs

        if self._encoding in ['simpleIndex', 'complex', 'lastPayload']:
            predicted = Flatten()(predicted)

        for _ in range(self._n_hidden_layers):
            predicted = Dense(self._n_hidden_units, activation=self._activation)(predicted)
            predicted = Dropout(self._dropout_rate)(predicted)

        if self._is_binary_classifier:
            predicted = Dense(1, activation='sigmoid')(predicted)
        else:
            predicted = Dense(self._encoding_parser.n_classes_y + 1, activation='softmax')(predicted)
        self._model = Model(model_inputs, predicted)

        if self._is_binary_classifier:
            self._model.compile(loss='binary_crossentropy', optimizer='adam')
        else:
            self._model.compile(loss='categorical_crossentropy', optimizer='adam')

        self._model.fit(train_data, targets, epochs=self._n_epochs)

    def predict(self, test_data: DataFrame) -> ndarray:
        """returns model predictions

        parses the encoded test dataset, then returns the model predictions

        :param test_data: encoded test dataset
        :return: model predictions

        """
        test_data = self._encoding_parser.parse_testing_dataset(test_data)

        predictions = self._model.predict(test_data)
        if self._is_binary_classifier:
            predictions = predictions.astype(bool)
        else:
            predictions = np.argmax(predictions, -1)
        return predictions

    def predict_proba(self, test_data: DataFrame) -> ndarray:
        """returns the classification probability

        parses the test dataset and returns the raw prediction probabilities of the model

        :param test_data: encoded test dataset
        :return: model prediction probabilities

        """
        test_data = self._encoding_parser.parse_testing_dataset(test_data)

        predictions = self._model.predict(test_data)
        if self._is_binary_classifier:
            predictions = np.max(predictions, -1)
            predictions = np.vstack((1 - predictions, predictions)).T
        return predictions

    def reset(self) -> None:
        """
        placeholder to allow use with other sklearn algorithms

        """


class NNRegressor(RegressorMixin):
    """
    Neural Network regressor, implements the same methods as the sklearn models to make it simple to add
    """

    # noinspection PyTypeChecker
    def __init__(self, **kwargs: Dict[str, Union[int, str, float]]):
        """initializes the Neural Network regressor

        :param kwargs: configuration containing the model parameters, encoding and training parameters

        """

        self._n_hidden_layers = int(kwargs['n_hidden_layers'])
        self._n_hidden_units = int(kwargs['n_hidden_units'])
        self._activation = str(kwargs['activation'])
        self._n_epochs = int(kwargs['n_epochs'])
        self._encoding = str(kwargs['encoding'])
        self._dropout_rate = float(kwargs['dropout_rate'])
        self._encoding_parser = EncodingParser(self._encoding, None, task=Tasks.REGRESSION)
        self._model = None

    def fit(self, train_data: DataFrame, targets: ndarray) -> None:
        """creates and fits the model

        first the encoded data is parsed, then the model created and then trained

        :param train_data: encoded training dataset
        :param targets: encoded target dataset

        """
        train_data = self._encoding_parser.parse_training_dataset(train_data)
        targets = self._encoding_parser.parse_targets(targets)

        model_inputs = Input(train_data.shape[1:])
        predicted = model_inputs

        if self._encoding in ['simpleIndex', 'complex', 'lastPayload']:
            predicted = Flatten()(predicted)

        for _ in range(self._n_hidden_layers):
            predicted = Dense(self._n_hidden_units, activation=self._activation)(predicted)
            predicted = Dropout(self._dropout_rate)(predicted)

        predicted = Dense(1, activation='sigmoid')(predicted)

        self._model = Model(model_inputs, predicted)
        self._model.compile(loss='mse', optimizer='adam')

        self._model.fit(train_data, targets, epochs=self._n_epochs)

    def predict(self, test_data: DataFrame) -> ndarray:
        """returns model predictions

        parses the encoded test dataset, then returns the model predictions

        :param test_data: encoded test dataset
        :return: model predictions

        """
        test_data = self._encoding_parser.parse_testing_dataset(test_data)

        predictions = self._model.predict(test_data)
        predictions = self._encoding_parser.denormalize_predictions(predictions)
        return predictions

    def reset(self) -> None:
        """
        placeholder to allow use with other sklearn algorithms

        """


class RNNTimeSeriesPredictor:
    """
    Recurrent Neural Network Time Series predictor, implements the same methods as the sklearn models to make it simple
    to add.
    This architecture is of the seq2seq type, taking as input a sequence (0...t) and outputting a sequence (1...t+1)
    """

    # noinspection PyTypeChecker
    def __init__(self, **kwargs: Dict[str, Union[int, str, float]]):
        """initializes the Recurrent Neural Network Time Series predictor

        :param kwargs: configuration containing the model parameters, encoding and training parameters

        """

        self._n_units = int(kwargs['n_units'])
        self._rnn_type = str(kwargs['rnn_type'])
        self._n_epochs = int(kwargs['n_epochs'])
        self._encoding = str(kwargs['encoding'])
        self._prefix_length = 0.25  # n x dataset length
        self._prediction_limit = 1.5  # n x dataset length
        self._encoding_parser = EncodingParser(self._encoding, None, task=Tasks.TIME_SERIES_PREDICTION)
        self._model = None

    def fit(self, train_data: DataFrame) -> None:
        """creates and fits the model

        first the encoded data is parsed, then the model created and then trained

        :param train_data: encoded training dataset

        """
        train_data = self._encoding_parser.parse_training_dataset(train_data)

        targets = train_data[:, 1:]
        train_data = train_data[:, :-1]

        model_inputs = Input(train_data.shape[1:])
        predicted = model_inputs

        predicted = Reshape((train_data.shape[1], train_data.shape[2] * train_data.shape[3]))(predicted)

        if self._rnn_type == 'lstm':
            predicted = LSTM(self._n_units, activation='relu', return_sequences=True)(predicted)
        elif self._rnn_type == 'gru':
            predicted = GRU(self._n_units, activation='relu', return_sequences=True)(predicted)

        predicted = Dense((self._encoding_parser.n_classes_x + 1) * self._encoding_parser.n_event_features)(predicted)
        predicted = Reshape((train_data.shape[1], train_data.shape[2], train_data.shape[3]))(predicted)

        predicted = Dense(train_data.shape[3], activation='softmax')(predicted)

        self._model = Model(model_inputs, predicted)
        self._model.compile(loss='categorical_crossentropy', optimizer='adam')
        self._model.fit(train_data, targets, epochs=self._n_epochs)

    def predict(self, test_data: DataFrame) -> ndarray:
        """returns model predictions

        parses the encoded test dataset, then returns the model predictions

        :param test_data: encoded test dataset
        :return: model predictions

        """
        test_data = self._encoding_parser.parse_testing_dataset(test_data)

        temp_prediction_length = test_data.shape[1] - 1
        temp_prediction = np.zeros((test_data.shape[0], temp_prediction_length, test_data.shape[2], test_data.shape[3]))

        final_prediction = test_data[:, :int(self._prefix_length * test_data.shape[1])]
        i = 0
        while True:
            temp_prediction_index = min(temp_prediction_length, final_prediction.shape[1])
            temp_prediction[:, :temp_prediction_index] = final_prediction[:, -temp_prediction_index:]

            model_predictions = self._model.predict(temp_prediction)

            next_step_prediction = model_predictions[:, temp_prediction_index - 1:temp_prediction_index, :]
            final_prediction = np.hstack((final_prediction, next_step_prediction))

            if 0 in next_step_prediction or i == int(self._prediction_limit * test_data.shape[1]):
                break
            i += 1

        final_prediction = np.argmax(final_prediction, -1)
        return final_prediction
