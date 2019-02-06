from typing import Dict, Union
from keras import Input, Model
from keras.layers import Flatten, Embedding, Dense, Dropout
from keras.utils import to_categorical
from pandas import DataFrame
from core.nn.encoding_parser import EncodingParser
from numpy import ndarray
import numpy as np


class NNClassifier:
    # noinspection PyTypeChecker
    def __init__(self, **kwargs: Dict[str, Union[int, str, float]]):
        self._n_hidden_layers = int(kwargs['n_hidden_layers'])
        self._n_hidden_units = int(kwargs['n_hidden_units'])
        self._activation = str(kwargs['activation'])
        self._n_epochs = int(kwargs['n_epochs'])
        self._encoding = str(kwargs['encoding'])
        self._dropout_rate = float(kwargs['dropout_rate'])
        self._is_binary_classifier = bool(kwargs['is_binary_classifier'])
        self._embedding_dim = 8  # TODO: add as parameter
        self._encoding_parser = EncodingParser(self._encoding, self._is_binary_classifier, regression_task=False)

    def fit(self, train_data: DataFrame, y: DataFrame) -> None:
        train_data = self._encoding_parser.parse_training_dataset(train_data)
        y = self._encoding_parser.parse_y(y)

        model_inputs = Input(train_data.shape[1:])
        predicted = model_inputs

        if self._encoding in ['simpleIndex', 'complex', 'lastPayload']:
            predicted = Embedding(self._encoding_parser.n_classes_x + 1, self._embedding_dim)(predicted)
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
        self._model.fit(train_data, y, epochs=self._n_epochs)

    def predict(self, test_data: DataFrame) -> ndarray:
        test_data = self._encoding_parser.parse_testing_dataset(test_data)

        predictions = self._model.predict(test_data)
        if self._is_binary_classifier:
            predictions = predictions.astype(bool)
        else:
            predictions = np.argmax(predictions, -1)
        return predictions

    def predict_proba(self, test_data: DataFrame) -> ndarray:
        test_data = self._encoding_parser.parse_testing_dataset(test_data)

        predictions = self._model.predict(test_data)
        if self._is_binary_classifier:
            predictions = np.max(predictions, -1)
            predictions = np.vstack((1-predictions, predictions)).T
        return predictions
