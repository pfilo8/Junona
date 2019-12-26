#!/bin/bash

python3 train_models.py --data data/creditcard.csv --config config/creditcard-config-1.json --n_iters 50
python3 train_models.py --data data/creditcard.csv --config config/creditcard-config-1.json --n_iters 50 --ratio 0.01
python3 train_models.py --data data/creditcard.csv --config config/creditcard-config-1.json --n_iters 50 --ratio 0.05
python3 train_models.py --data data/creditcard.csv --config config/creditcard-config-1.json --n_iters 50 --ratio 0.1
python3 train_models.py --data data/creditcard.csv --config config/creditcard-config-1.json --n_iters 50 --ratio 0.2
python3 train_models.py --data data/creditcard.csv --config config/creditcard-config-1.json --n_iters 50 --ratio 0.5
