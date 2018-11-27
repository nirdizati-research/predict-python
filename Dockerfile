FROM python:3.6-stretch

RUN apt-get update && \
    apt-get install -y git-core python3-numpy python3-scipy && \
    rm -rf /var/lib/apt/lists/*

# Add requirements file used by pip install
ADD ./requirements.txt /predict-python/
WORKDIR /predict-python

# Run pip install to install all python dependenies
RUN pip3 install -r requirements.txt && \
    pip3 install git+git://github.com/scikit-multiflow/scikit-multiflow.git#egg=scikit-multiflow

# Add all the project files
ADD . /predict-python

EXPOSE 8000
