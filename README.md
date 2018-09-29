# Predict python

[![Build Status](https://travis-ci.org/nirdizati-research/predict-python.svg?branch=master)](https://travis-ci.org/TKasekamp/predict-python)

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

Start jobs from command line
```commandline
curl --request POST \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "type": "classification",
    "split_id": 1,
    "config": {
      "encodings": ["simpleIndex"],
      "clusterings": ["noCluster"],
      "methods": ["randomForest"],
      "label": {"type": "remaining_time"},
      "encoding": {"prefix_length": 3, "generation_type": "only", "padding": "zero_padding"}
    }
  }' \
http://localhost:8000/jobs/multiple
```

Creating a single split options.

* $SPLIT_TYPE has to be one of `split_sequential`, `split_random`, `split_temporal`, `split_strict_temporal`. By default `split_sequential`.
* `test_size` has to be from 0 to 1. By default 0.2
```commandline
curl --request POST \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "type": "single",
    "original_log": 1, 
    "config": {
      "test_size": 0.2,
      "split_type": $SPLIT_TYPE
    }
  }' \
http://localhost:8000/splits/
```

#### Advanced configuration

Prediction methods accept configuration for sklearn classification/regression methods. 
The Job config must contain a dict with only the supported options for that method. 
The dict name must take the format "type.method". For classification randomForest this would be `classification.randomForest`.
Advanced configuration is optional. Look at `jobs/job_creator.py` for default values.

For example, the configuration for classification KNN would have to be like:

```commandline
curl --request POST \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "type": "classification",
    "split_id": 1,
    "config": {
      "encodings": ["simpleIndex"],
      "clusterings": ["noCluster"],
      "methods": ["knn"],
      "classification.knn": {
        "n_neighbors": 5,
        "weights": "uniform"
      },
      "label": {"type": "remaining_time"},
      "encoding": {"prefix_length": 3, "generation_type": "up_to", "padding": "no_padding"}
    }
  }' \
http://localhost:8000/jobs/multiple
```

## Labelling job
Log encoding and labelling can be tested before prediction. It supports all the same values as classification and 
regression jobs but the method and clustering.

```commandline
curl --request POST \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "type": "labelling",
    "split_id": 5,
    "config": {
      "label": {"type": "remaining_time"},
      "encoding": {"prefix_length": 3, "generation_type": "up_to", "padding": "no_padding"}
    }
  }' \
http://localhost:8000/jobs/multiple
```

# Running in a new environment
The following is all the commands needed to set up the backend and [predict-react](https://github.com/nirdizati-research/predict-react) in a new environment. The guide was created for Ubuntu 18.04, but it should work on any linux or mac system. 

```bash
sudo apt install git
sudo apt install curl
sudo apt install make
sudo apt-get install build-essential # maybe not everything is needed, but at least g++

# for npm
curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash -
sudo apt-get install -y nodejs

# Redis for job queues
# https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-18-04
sudo apt install redis-server
sudo nano /etc/redis/redis.conf
# Follow instructions and change supervised setting
sudo systemctl restart redis.service

# Frontend
git clone https://github.com/nirdizati-research/predict-react.git
npm install
npm start # test that is available in localhost:3000

# Python 3 needed
cd .. # back to home folder
sudo apt install python3-pip
git clone https://github.com/nirdizati-research/predict-python.git
pip3 install -r requirements.txt

# DB setup
python3 manage.py migrate
python3 manage.py loaddata all_model_data.json
# deployment
chmod +x deployment.sh killall.sh
# Change .deployment.sh to run at port 8000. Change 
# sudo nohup python3 manage.py runserver 0.0.0.0:80 &
# to
# sudo nohup python3 manage.py runserver 0.0.0.0:8000 & 


# Frontend start. run this in predict-react. server runs until command exited
npm start
# Frontend visible at localhost:3000
# Backend start. Run in predict-python. Runs as a background process until ./killall.sh is run
./deployment.sh
# Backend visible at localhost:8000
```

## Contributors
- [@TKasekamp](https://github.com/TKasekamp) Tõnis Kasekamp 
- [@stebranchi](https://github.com/stebranchi) Stefano Branchi
