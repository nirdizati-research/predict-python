#!/usr/bin/env bash

for folder in core encoders jobs logs nirdizati-research pred_models runtime utils:
do
    sphinx-apidoc -f -o source/ ../$folder
done
