import sys
import glob
import numpy as np
import pandas as pd


def read_dataframes(file_pattern):
    return pd.concat([pd.read_csv(file) for file in glob.glob(file_pattern)], sort=False)


def aggregate_score(dataframe, func, name_column='Name'):
    return dataframe.groupby(name_column).agg(func).reset_index()


if __name__ == '__main__':
    dir_path = sys.argv[1]

    all_dataframes = read_dataframes(f'{dir_path}*.csv')

    all_dataframes.to_csv(f'{dir_path}results_aggregated.csv', index=False)
    aggregate_score(all_dataframes, np.mean).to_csv(f'{dir_path}results_mean.csv', index=False)
    aggregate_score(all_dataframes, np.std).to_csv(f'{dir_path}results_std.csv', index=False)
