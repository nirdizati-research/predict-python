from typing import Dict, Union

from keras import Input, Model
from keras.layers import Flatten, Dense, Dropout
from numpy import ndarray
from pandas import DataFrame
from sklearn.base import RegressorMixin

from src.encoding.encoding_parser import EncodingParser
from src.predictive_model.models import PredictiveModels


class NNRegressor(RegressorMixin):
    """
    Neural Network regressor, implements the same methods as the sklearn models to make it simple to add
    """

    # noinspection PyTypeChecker
    def __init__(self, **kwargs: Dict[str, Union[int, str, float]]):
        """initializes the Neural Network regressor

        :param kwargs: configuration containing the predictive_model parameters, encoding and training parameters

        """

        self._n_hidden_layers = int(kwargs['n_hidden_layers'])
        self._n_hidden_units = int(kwargs['n_hidden_units'])
        self._activation = str(kwargs['activation'])
        self._n_epochs = int(kwargs['n_epochs'])
        self._encoding = str(kwargs['encoding'])
        self._dropout_rate = float(kwargs['dropout_rate'])
        self._encoding_parser = EncodingParser(self._encoding, None, task=PredictiveModels.REGRESSION.value)
        self._model = None

    def fit(self, train_data: DataFrame, targets: ndarray) -> None:
        """creates and fits the predictive_model

        first the encoded data is parsed, then the predictive_model created and then trained

        :param train_data: encoded training dataset
        :param targets: encoded target dataset

        """
        targets = DataFrame(targets, columns=['label'])

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
        """returns predictive_model predictions

        parses the encoded test dataset, then returns the predictive_model predictions

        :param test_data: encoded test dataset
        :return: predictive_model predictions

        """
        test_data = self._encoding_parser.parse_testing_dataset(test_data)

        predictions = self._model.predict(test_data)
        predictions = self._encoding_parser.denormalize_predictions(predictions)
        return predictions

    def reset(self) -> None:
        """
        placeholder to allow use with other sklearn algorithms

        """
