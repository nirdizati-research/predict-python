# Predict python

[![License MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![HitCount](http://hits.dwyl.io/nirdizati-research/predict-python.svg)](http://hits.dwyl.io/nirdizati-research/predict-python)

Master

[![Build Status](https://travis-ci.org/nirdizati-research/predict-python.svg?branch=master)](https://travis-ci.org/nirdizati-research/predict-python)
[![codecov](https://codecov.io/gh/nirdizati-research/predict-python/branch/master/graph/badge.svg)](https://codecov.io/gh/nirdizati-research/predict-python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/nirdizati-research/predict-python.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/nirdizati-research/predict-python/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/nirdizati-research/predict-python.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/nirdizati-research/predict-python/context:python)

Development

[![Build Status](https://travis-ci.org/nirdizati-research/predict-python.svg?branch=development)](https://travis-ci.org/nirdizati-research/predict-python)
[![codecov](https://codecov.io/gh/nirdizati-research/predict-python/branch/development/graph/badge.svg)](https://codecov.io/gh/nirdizati-research/predict-python)


Django backend server for machine learning on event logs.

## Running in a new environment
The docker build is available @ https://hub.docker.com/r/nirdizatiresearch/predict-python/ in any case if you prefer to setup your environment on your own you can refer the [Dockerfile](Dockerfile).

## Docker Compose

On first run to setup the database, you can run:
```commandline
docker-compose run server python manage.py migrate
```

To run the project:
```commandline
docker-compose up redis server scheduler worker
```

To access a generic remote Django server you can use the ssh tunneling functionality as shown in the following sample:
```commandline
ssh -L 8000:127.0.0.1:8000 <user>@<host>
```

## Run an instance of the project
If you are familiar with docker-compose the [docker-compose](docker-compose.yml) file is available, otherwise if you use PyCharm as IDE run the provided configurations.

Finally, from the command line you can use the following sample commands to interact with our software.

Start server with
```commandline
python manage.py runserver
```

Run tests with one of the following
```commandline
python manage.py test
./manage.py test
```

NB: always run a redis-server in background if you want your server to accept any incoming post requests!

Start by running migrations and adding sample data
```commandline
python manage.py migrate
python manage.py loaddata <your_file.json>
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

## Documentation
This project allows documentation to be built automatically using [sphinx](http://www.sphinx-doc.org/en/master/). All
the documentation-related files are in the [docs/](docs) folder, structured as:

```bash
└── docs/
    ├── build/
    │   ├── doctrees/
    │   └── html/
    ├── source
    │    ├── _static/
    │    ├── _templates/
    │    ├── api/
    │    ├── readme/
    │    ├── conf.py
    │    └── index.rst
    ├── generate_modules.sh
    └── Makefile
``` 

in the [html/](docs/html) the built html files are placed, whereas in the [source/](docs/source) there are all the source
files. The [_static/](docs/source/_static) contains the images used in the final html files, as the logo: place eventual 
screenshots etc. here. The [api/](docs/source/api) contains all the files used for automatically fetching docstrings in
the project, you shouldn't edit them as they are all replaced when re-building the documentation. The [readme/](docs/source/readme)
folder contains the .rst copies of the readmes used, when updating the project's readme, please also update those accordingly.
The [conf.py](docs/source/conf.py) contains all the sphinx settings, along with the theme used (sphinx-rtd-theme).

The [index.rst](docs/source/index.rst) file is the documentation entry point, change this file to change the main documentation
structure as needed. After updating the docstrings in the project, please re-run the [generate_modules.sh](docs/generate_modules.sh) script,
that simply uses the ```sphinx-apidoc``` command to re-create the api .rst files.

Finally, the [Makefile](docs/Makefile) is used when building the entire documentation, please run a ```make clean```
```make html``` when you want updated docs.

To summarize, after changing docstrings or the readme.rst files, simply run:

```bash
sh generate_modules.sh
make clean
make html
```

Documentation is also hosted on [readthedocs.com](https://nirdizati-research.readthedocs.io/en/development/) and built
automatically after each commit on the master or development branch, make sure to have the api files updated in advance.
### Note on CUDA enabled systems
As this project detects when a compatible GPU is present in the system and tries to use it, please add a 
```CUDA_VISIBLE_DEVICES=0``` flag as an environment variable if you encounter problems.


## Contributors
- [@stebranchi](https://github.com/stebranchi) Stefano Branchi
- [@dfmchiara](https://github.com/dfmchiara) Chiara Di Francescomarino 
- [@TKasekamp](https://github.com/TKasekamp) Tõnis Kasekamp 
- [@mrsonuk](https://github.com/mrsonuk) Santosh Kumar
- [@fmmaggi](https://github.com/fmmaggi) Fabrizio Maggi
- [@WilliamsRizzi](https://github.com/WilliamsRizzi) Williams Rizzi
- [@HitLuca](https://github.com/HitLuca) Luca Simonetto
