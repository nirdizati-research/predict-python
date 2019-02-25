#!/usr/bin/env bash

rm source/api/*
for folder in core encoders jobs logs nirdizati-research pred_models runtime utils
do
    sphinx-apidoc -e -f -o source/api/ ../${folder}
done
