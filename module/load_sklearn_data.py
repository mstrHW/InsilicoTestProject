from sklearn import datasets
import pandas as pd
import numpy as np

from definitions import *
import matplotlib.pyplot as plt


def load_data():
    data = datasets.load_iris()
    concatenated = np.c_[data['data'], data['target']]
    feature_names = data['feature_names']
    target_name = 'target'
    data_df = pd.DataFrame(data=concatenated, columns=feature_names + [target_name])
    print(data_df.dtypes)
    print(data_df.shape)
    print(data_df.columns)
    return data_df, feature_names, target_name


def get_nans_percent(df: pd.DataFrame, columns: List[str] = None) -> float:
    target_columns = df.columns.tolist()
    if isinstance(columns, list):
        target_columns = columns
    if isinstance(columns, str):
        target_columns = [columns]
    nans_count = get_nans_count(df, target_columns)
    df_count = 0
    for column in target_columns:
        df_count += df[column].shape[0]
    return nans_count/df_count


def get_nans_count(data_frame: pd.DataFrame, columns: List[str] = None, by_columns: bool = False) -> int:
    nans_summ = 0
    target_columns = data_frame.columns.tolist()
    if columns:
        target_columns = columns

    if by_columns:
        nans_summ += data_frame[target_columns].isna().sum()
    else:
        nans_summ += data_frame[target_columns].isna().sum().sum()
    return nans_summ
