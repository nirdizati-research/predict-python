from typing import Dict, Union

import numpy as np
from keras import Input, Model
from keras.layers import Dense, LSTM, GRU, Reshape
from keras.utils import to_categorical
from numpy import ndarray
from pandas import DataFrame

from encoders.encoding_parser import EncodingParser, Tasks


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
        self._prefix_length = 4
        self._prediction_limit = 10
        self._embedding_dim = 8  # TODO: add as parameter
        self._encoding_parser = EncodingParser(self._encoding, None, task=Tasks.TIME_SERIES_PREDICTION)
        self._model = None

    def fit(self, train_data: DataFrame) -> None:
        """creates and fits the model

        first the encoded data is parsed, then the model created and then trained

        :param train_data: encoded training dataset

        """
        train_data = self._encoding_parser.parse_training_dataset(train_data)
        train_data = to_categorical(train_data, self._encoding_parser.n_classes_x + 1)

        if self._encoding == 'simpleIndex':
            train_data = np.expand_dims(train_data, -2)

        y = train_data[:, 1:]
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
        self._model.fit(train_data, y, epochs=self._n_epochs)

    def predict(self, test_data: DataFrame) -> ndarray:
        """returns model predictions

        parses the encoded test dataset, then returns the model predictions

        :param test_data: encoded test dataset
        :return: model predictions

        """
        test_data = self._encoding_parser.parse_testing_dataset(test_data)
        test_data = to_categorical(test_data, self._encoding_parser.n_classes_x + 1)

        if self._encoding == 'simpleIndex':
            test_data = np.expand_dims(test_data, -2)

        temp_prediction_length = test_data.shape[1] - 1
        temp_prediction = np.zeros((test_data.shape[0], temp_prediction_length, test_data.shape[2], test_data.shape[3]))

        final_prediction = test_data[:, :self._prefix_length]
        i = 0
        while True:
            temp_prediction_index = min(temp_prediction_length, final_prediction.shape[1])
            temp_prediction[:, :temp_prediction_index] = final_prediction[:, -temp_prediction_index:]

            model_predictions = self._model.predict(temp_prediction)

            next_step_prediction = model_predictions[:, temp_prediction_index - 1:temp_prediction_index, :]
            final_prediction = np.hstack((final_prediction, next_step_prediction))

            if 0 in next_step_prediction or i == self._prediction_limit:
                break
            i += 1

        final_prediction = np.argmax(final_prediction, -1)
        return final_prediction
