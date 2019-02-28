#!/usr/bin/env bash

rm -r source/api/*

sphinx-apidoc -f -o source/api/ ../src
