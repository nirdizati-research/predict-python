FROM python:3.6-stretch

RUN apt-get update && \
    apt-get install -y git-core python3-numpy python3-scipy && \
    rm -rf /var/lib/apt/lists/*

ADD . /predict-python
WORKDIR /predict-python
RUN ls

RUN pip3 install -r requirements.txt && \
    pip3 install git+git://github.com/scikit-multiflow/scikit-multiflow.git#egg=scikit-multiflow

RUN python3 manage.py migrate

#CMD [ "python", "manage.py", "runserver", "localhost:80" ]
#CMD [ "python", "manage.py", "rqscheduler", "localhost:80" ]
#CMD [ "python", "manage.py", "rqworker", "default" ]