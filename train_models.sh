#!/bin/bash

python train_models.py --data data/creditcard.csv --config config/creditcard-config-100.json --n_iters 50
python train_models.py --data data/creditcard.csv --config config/creditcard-config-200.json --n_iters 50
