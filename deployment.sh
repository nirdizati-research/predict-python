#!/bin/bash
sudo pkill -f python
# Server has both python and python3
sudo nohup python3 manage.py runserver 0.0.0.0:80 &
sudo nohup python3 manage.py rqworker default &
sudo nohup python3 manage.py rqworker default &