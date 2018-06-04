#!/bin/sh
sudo lsof -t -i tcp:80 -s tcp:listen | sudo xargs kill
sudo pkill -f python