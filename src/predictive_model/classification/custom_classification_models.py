from typing import Dict, Union

import numpy as np
from keras import Input, Model
from keras.layers import Flatten, Dense, Dropout
from numpy import ndarray
from pandas import DataFrame
from sklearn.base import ClassifierMixin

from src.encoding.encoding_parser import EncodingParser, Tasks


class NNClassifier(ClassifierMixin):
    """
    Neural Network classifier, implements the same methods as the sklearn models to make it simple to add
    """

    # noinspection PyTypeChecker
    def __init__(self, **kwargs: Dict[str, Union[int, str, float]]):
        """initializes the Neural Network classifier

        :param kwargs: configuration containing the predictive_model parameters, encoding and training parameters

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
        """creates and fits the predictive_model

        first the encoded data is parsed, then the predictive_model created and then trained

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
        """returns predictive_model predictions

        parses the encoded test dataset, then returns the predictive_model predictions

        :param test_data: encoded test dataset
        :return: predictive_model predictions

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

        parses the test dataset and returns the raw prediction probabilities of the predictive_model

        :param test_data: encoded test dataset
        :return: predictive_model prediction probabilities

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
