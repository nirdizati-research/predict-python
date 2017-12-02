# Predict python

[![Build Status](https://travis-ci.org/TKasekamp/predict-python.svg?branch=master)](https://travis-ci.org/TKasekamp/predict-python)

Django backend server for machine learning on event logs.

## Requirements
Click the CI badge to see the supported Python versions.

## Setup

Install with
```commandline
pip install -r requirements.txt
```

Start server with
```commandline
python manage.py runserver
```

Run tests with one of the following
```commandline
python manage.py test
./manage.py test
```

## Useful database operations
Start by running migrations and adding sample data
```commandline
python manage.py migrate
python manage.py loaddata all_model_data.json
```