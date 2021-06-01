def time_series_prediction_rnn():
    """"
    Returns the default parameters for time series prediction RNN
    """
    return {
        'n_units': 16,
        'rnn_type': 'lstm',
        'n_epochs': 10,
    }
