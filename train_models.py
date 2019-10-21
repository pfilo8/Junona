import os
import warnings

import numpy as np
import pandas as pd
import multiprocessing as mp

from datetime import datetime
from sklearn.model_selection import train_test_split

from src.utils import get_script_args, load_json, dict_union
from src.utils import create_cost_matrix, create_X_y
from src.utils import generate_models, generate_summary
from src.utils import train_standard_models, train_to_models, train_bmr_models

RANDOM_STATE = 42
OUTPUT_DIR = 'outputs'
CURRENT_OUTPUT_DIR = 'results' + datetime.now().isoformat()

warnings.filterwarnings('ignore')
# np.random.seed(RANDOM_STATE)


def train_iteration(i):
    print(f"Iteration: {i}.")
    models = generate_models()

    X_train, X_test, y_train, y_test, cost_matrix_train, cost_matrix_test = train_test_split(
        X, y, cost_matrix, train_size=0.5, stratify=y  # , random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test, cost_matrix_val, cost_matrix_test = train_test_split(
        X_test, y_test, cost_matrix_test, train_size=0.33, stratify=y_test  # , random_state=RANDOM_STATE
    )

    standard_models = train_standard_models(X_train, y_train, cost_matrix_train, X_val, y_val, models)
    threshold_optimized_models = train_to_models(X_val, y_val, cost_matrix_val, standard_models)
    bmr_models = train_bmr_models(X_val, y_val, standard_models)

    trained_models = dict_union(standard_models, threshold_optimized_models, bmr_models)
    results = generate_summary(trained_models, X_test, y_test, cost_matrix_test)

    filename = str(i) + '.csv'
    filepath = os.path.join(OUTPUT_DIR, CURRENT_OUTPUT_DIR, filename)

    results.to_csv(filepath, index=False)


if __name__ == '__main__':
    args = get_script_args()

    config = load_json(args['config'])
    df = pd.read_csv(args['data'])
    X, y = create_X_y(
        data=df,
        class_column=config['ClassColumn'],
        drop_columns=config['DropColumns']
    )
    cost_matrix = create_cost_matrix(
        data=df,
        fp_cost=config['FalsePositiveCost'],
        fn_cost=config['FalseNegativeCost'],
        tp_cost=config['TruePositiveCost'],
        tn_cost=config['TrueNegativeCost']
    )

    os.mkdir(os.path.join(OUTPUT_DIR, CURRENT_OUTPUT_DIR))

    pool = mp.Pool(mp.cpu_count() - 1)
    results = pool.map(train_iteration, range(int(args["n_iters"])))
