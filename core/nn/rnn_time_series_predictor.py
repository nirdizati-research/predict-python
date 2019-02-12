from typing import Dict, Union

import numpy as np
from keras import Input, Model
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.utils import to_categorical
from numpy import ndarray
from pandas import DataFrame

from .encoding_parser import EncodingParser


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
        self._embedding_dim = 8  # TODO: add as parameter
        self._encoding_parser = EncodingParser(self._encoding, None, regression_task=True)
        self._model = None

    def fit(self, train_data: DataFrame) -> None:
        """creates and fits the model

        first the encoded data is parsed, then the model created and then trained
        :param train_data: encoded training dataset
        """
        train_data = self._encoding_parser.parse_training_dataset(train_data)

        y = to_categorical(train_data[:, 1:], self._encoding_parser.n_classes_x + 1)
        train_data = train_data[:, :-1]

        model_inputs = Input(train_data.shape[1:])
        predicted = model_inputs

        predicted = Embedding(self._encoding_parser.n_classes_x + 1, self._embedding_dim)(predicted)

        if self._rnn_type == 'lstm':
            predicted = LSTM(self._n_units, activation='relu', return_sequences=True)(predicted)
        elif self._rnn_type == 'gru':
            predicted = GRU(self._n_units, activation='relu', return_sequences=True)(predicted)

        predicted = Dense(self._encoding_parser.n_classes_x + 1, activation='softmax')(predicted)

        self._model = Model(model_inputs, predicted)
        self._model.compile(loss='categorical_crossentropy', optimizer='adam')
        self._model.fit(train_data, y, epochs=self._n_epochs)

    def predict(self, test_data: DataFrame) -> ndarray:
        """returns model predictions

        parses the encoded test dataset, then returns the model predictions
        :param test_data: encoded test dataset
        :return: model predictions
        """
        test_data = self._encoding_parser.parse_testing_dataset(test_data)[:, :-1]
        predictions = self._model.predict(test_data)
        return np.argmax(predictions, -1)
