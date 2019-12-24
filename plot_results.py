import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update(
    {
        'font.size': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.titlesize': 22,
        'axes.titleweight': 'normal'
    }
)


def read_dataframes(file_pattern):
    return pd.concat([pd.read_csv(file) for file in glob.glob(file_pattern)], sort=False)


def prepare_data(dataframe):
    df = dataframe.copy()
    df[['Model class', 'Model', 'Extra model']] = pd.DataFrame(df.Name.str.split('-').values.tolist())
    df['Model'] = df['Model'].replace({'CostSensitiveLogisticRegression': 'CS Logistic Regression',
                                       'CostSensitiveDecisionTreeClassifier': 'CS Decistion Tree'})
    df['Extra model'] = df['Extra model'].fillna('Standard')
    return df


def generate_plot(data, input_dir_name, type=None, ):
    filename = f'{input_dir_name}-{type}.png'
    output_path = os.path.join('outputs', '200_plots', filename)

    color_mapping = {'Standard': 'k', 'BMR': 'r', 'TO': 'b'}
    hue_order = ['Standard', 'TO', 'BMR']

    fig, ax = plt.subplots(figsize=(17, 12))

    sns.barplot(x=type, y='Model', data=data, hue='Extra model',
                ax=ax, alpha=0.6, palette=color_mapping, hue_order=hue_order)

    ax.set_xlim([0, 1])
    plt.title(f'Results of experiment - {type}')
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == '__main__':
    input_dir = sys.argv[1]
    input_dir_name = os.path.split(input_dir)[-1]

    df = read_dataframes(os.path.join(input_dir, '*.csv'))
    df = prepare_data(df)
    generate_plot(df, input_dir_name, type='Savings')
    generate_plot(df, input_dir_name, type='F1')
