from typing import Dict, Union

from keras import Input, Model
from keras.layers import Dense, Embedding, Flatten, Dropout
from numpy import ndarray
from pandas import DataFrame

from .encoding_parser import EncodingParser


class NNRegressor:
    # noinspection PyTypeChecker
    def __init__(self, **kwargs: Dict[str, Union[int, str, float]]):
        self._n_hidden_layers = int(kwargs['n_hidden_layers'])
        self._n_hidden_units = int(kwargs['n_hidden_units'])
        self._activation = str(kwargs['activation'])
        self._n_epochs = int(kwargs['n_epochs'])
        self._encoding = str(kwargs['encoding'])
        self._dropout_rate = float(kwargs['dropout_rate'])
        self._embedding_dim = 8  # TODO: add as parameter
        self._encoding_parser = EncodingParser(self._encoding, None, regression_task=True)
        self._model = None

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

        predicted = Dense(1, activation='sigmoid')(predicted)

        self._model = Model(model_inputs, predicted)
        self._model.compile(loss='mse', optimizer='adam')
        self._model.fit(train_data, y, epochs=self._n_epochs)

    def predict(self, test_data: DataFrame) -> ndarray:
        test_data = self._encoding_parser.parse_testing_dataset(test_data)

        predictions = self._model.predict(test_data)
        predictions = self._encoding_parser.denormalize_predictions(predictions)
        return predictions
