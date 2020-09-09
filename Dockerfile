FROM python:3.6-stretch

#Install git & graphviz
RUN apt-get update \
    && apt-get install -y git \
    && apt-get install -y graphviz

#Downgrade pip until https://github.com/oracle/Skater/issues/292 is closed
RUN python3 -m pip install --force-reinstall pip==20.1

# Add requirements file used by pip install
ADD ./requirements.txt /predict-python/
WORKDIR /predict-python

# Run pip install to install all python dependenies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir git+git://github.com/scikit-multiflow/scikit-multiflow.git#egg=scikit-multiflow
#RUN pip3 install --no-cache-dir git+git://github.com/oracle/Skater.git@a945bac6ed43c9c46230985b9cea1f08e0860cdf

# Add all the project files
ADD . /predict-python

EXPOSE 8000
