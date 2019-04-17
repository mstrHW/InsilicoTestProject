from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    return scaler.fit_transform(df.values)


def binarize_target(y):
    lb = LabelBinarizer()
    lb.fit(y)
    return lb.transform(y)


def split_dataset(data_frame: pd.DataFrame, test_size: float) -> [pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(data_frame, test_size=test_size)
    return train, test
