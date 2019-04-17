from sklearn import datasets
import pandas as pd
import numpy as np

from definitions import *


def choose_dataset_dict() -> Dict:
    choose_model = {
        'iris': load_iris,
    }
    return choose_model


def possible_datasets() -> List[str]:
    return list(choose_dataset_dict().keys())


def load_dataset(dataset_name: str):
    choose_dataset = choose_dataset_dict()
    return choose_dataset[dataset_name]()


def load_iris() -> [pd.DataFrame, List[str], List[str]]:
    data = datasets.load_iris()
    concatenated = np.c_[data['data'], data['target']]
    feature_names = data['feature_names']
    target_name = ['target']
    data_df = pd.DataFrame(data=concatenated, columns=feature_names + target_name)
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
