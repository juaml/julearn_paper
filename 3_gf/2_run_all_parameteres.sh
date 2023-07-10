#!/bin/bash

# Warning, this script will take a long time to run
# It is merely meant as an example of all the different parameter combinations that we ran

python 1_predict_gf.py 0.01 pos Repeated10Fold
python 1_predict_gf.py 0.05 pos Repeated10Fold
python 1_predict_gf.py 0.10 pos Repeated10Fold
python 1_predict_gf.py 0.01 neg Repeated10Fold
python 1_predict_gf.py 0.05 neg Repeated10Fold
python 1_predict_gf.py 0.10 neg Repeated10Fold
python 1_predict_gf.py 0.01 posneg Repeated10Fold
python 1_predict_gf.py 0.05 posneg Repeated10Fold
python 1_predict_gf.py 0.10 posneg Repeated10Fold
python 1_predict_gf.py 0.01 pos LOO