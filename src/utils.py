import json
import argparse
import xgboost

import numpy as np
import pandas as pd

from itertools import chain

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from costcla.metrics import cost_loss, savings_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from costcla.models import BayesMinimumRiskClassifier, ThresholdingOptimization
from costcla.models import CostSensitiveDecisionTreeClassifier, CostSensitiveLogisticRegression
from costcla.models import CostSensitiveRandomForestClassifier, CostSensitiveBaggingClassifier, \
    CostSensitivePastingClassifier, CostSensitiveRandomPatchesClassifier

N_JOBS = -1
RANDOM_STATE = 42

CI_MODELS = [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]
CST_MODELS = [CostSensitiveLogisticRegression, CostSensitiveDecisionTreeClassifier]
XGB_MODELS = [xgboost.XGBClassifier]
ECSDT_MODELS = [CostSensitiveRandomForestClassifier, CostSensitiveBaggingClassifier,
                CostSensitivePastingClassifier, CostSensitiveRandomPatchesClassifier]


def dict_union(*args):
    return dict(chain.from_iterable(d.items() for d in args))


def get_script_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="Path to data.")
    ap.add_argument("-c", "--config", required=True, help="Path to config.")
    ap.add_argument("-n", "--n_iters", required=False, default=10, help="Number of MC iterations.")
    args = vars(ap.parse_args())
    return args


def load_json(path):
    with open(path) as f:
        config = json.load(f)
    return config


def create_cost_matrix(data, fp_cost, fn_cost, tp_cost, tn_cost):
    # false positives, false negatives, true positives, true negatives
    def generate_cost(dataframe, cost):
        return dataframe[cost] if type(cost) == str else cost

    cost_matrix = np.zeros((data.shape[0], 4))

    cost_matrix[:, 0] = generate_cost(data, fp_cost)
    cost_matrix[:, 1] = generate_cost(data, fn_cost)
    cost_matrix[:, 2] = generate_cost(data, tp_cost)
    cost_matrix[:, 3] = generate_cost(data, tn_cost)

    return cost_matrix


def create_X_y(data, class_column, drop_columns):
    X = data.drop(drop_columns + [class_column], axis=1)
    y = data[class_column]
    return X, y


def _create_bmr_model(model, X_val, y_val, calibration=True):
    y_hat_val_proba = model.predict_proba(X_val)

    bmr = BayesMinimumRiskClassifier(calibration=calibration)
    bmr.fit(y_val, y_hat_val_proba)

    return model, bmr


def _create_threshold_optimized_model(model, X_val, y_val, cost_matrix_val, calibration=True):
    y_hat_val_proba = model.predict_proba(X_val)

    threshold_opt = ThresholdingOptimization(calibration=calibration)
    threshold_opt.fit(y_hat_val_proba, cost_matrix_val, y_val)

    return model, threshold_opt


def generate_models():
    models = {
        'CI-LogisticRegression': LogisticRegression(),
        'CI-DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'CI-RandomForest': RandomForestClassifier(random_state=RANDOM_STATE),
        'CI-XGBoost': xgboost.XGBClassifier(random_state=RANDOM_STATE, verbosity=0),
        'CST-CostSensitiveDecisionTreeClassifier': CostSensitiveDecisionTreeClassifier()
    }
    return models


def train_standard_models(X_train, y_train, cost_matrix_train, X_val, y_val, models):
    trained_models = models.copy()
    # Standard model training
    for model in trained_models.values():
        model_type = type(model)
        print(model_type)
        if model_type in XGB_MODELS:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val), (X_train, y_train)],
                eval_metric='aucpr',
                early_stopping_rounds=50,
                verbose=False
            )
        elif model_type in CST_MODELS or model_type in ECSDT_MODELS:
            model.fit(X_train, y_train, cost_matrix_train)
        elif model_type in CI_MODELS:
            model.fit(X_train, y_train)
        else:
            raise ValueError(f'Unknown model type: {model_type}.')

    return trained_models


def train_to_models(X_val, y_val, cost_matrix_val, models):
    trained_models = {}

    for name, model in models.items():
        model_type = type(model)
        if model_type in XGB_MODELS or model_type in CI_MODELS:
            for calibration in [True]:
                print(model_type, calibration)
                to_model = _create_threshold_optimized_model(model, X_val, y_val, cost_matrix_val, calibration=calibration)
                model_name = name + '-TO' #+ f'_{calibration}'
                trained_models[model_name] = to_model

    return trained_models


def train_bmr_models(X_val, y_val, models):
    trained_models = {}

    for name, model in models.items():
        model_type = type(model)
        print(model_type)
        if model_type in XGB_MODELS or model_type in CI_MODELS:
            for calibration in [True]:
                print(calibration)
                to_model = _create_bmr_model(model, X_val, y_val, calibration=calibration)
                model_name = name + '-BMR' #+ f'_{calibration}'
                trained_models[model_name] = to_model

    return trained_models


def _create_model_summary(model, name, X_test, y_test, cost_matrix_test):
    standard_model_type = type(model)
    if standard_model_type == tuple:
        standard_model, extra_model = model
        extra_model_type = type(extra_model)
        if extra_model_type == BayesMinimumRiskClassifier:
            y_hat_proba = standard_model.predict_proba(X_test)
            y_hat = extra_model.predict(y_hat_proba, cost_matrix_test)
        elif extra_model_type == ThresholdingOptimization:
            y_hat_proba = standard_model.predict_proba(X_test)
            y_hat = extra_model.predict(y_hat_proba)
        else:
            raise ValueError(f'Unknown model type: {extra_model_type}.')
    elif standard_model_type in ECSDT_MODELS:
        y_hat = model.predict(X_test, cost_matrix_test)
    else:
        y_hat = model.predict(X_test)
    return {
        'Name': name,
        'Accuracy': accuracy_score(y_test, y_hat),
        'Precision': precision_score(y_test, y_hat),
        'Recall': recall_score(y_test, y_hat),
        'F1': f1_score(y_test, y_hat),
        'Cost': cost_loss(y_test, y_hat, cost_matrix_test),
        'Savings': savings_score(y_test, y_hat, cost_matrix_test)
    }


def generate_summary(trained_models, X_test, y_test, cost_matrix_test):
    results = pd.DataFrame(
        [_create_model_summary(
            model,
            name,
            X_test,
            y_test,
            cost_matrix_test
        ) for name, model in trained_models.items()
        ]
    )
    return results
