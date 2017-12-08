#!/bin/bash

wget -O hw4_model.h5 'https://www.dropbox.com/s/1pihyrdm4ja1fpw/hw4_model.h5?dl=1'
python3 hw4_predict.py $@ $@
