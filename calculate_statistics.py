import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

from module.mongodb_loader import pickle, df_to_json, ax_to_json, dict_to_json


def np_get_distribution(x):
    sns.kdeplot(x, shade=True)    # Todo: return distribution
    # plt.show()


def pd_get_distribution(feature):
    return feature.plot.kde()


def np_get_percentiles(feature, required_percentiles):
    percentiles = dict()
    for percentile in required_percentiles:
        percentiles[percentile] = np.percentile(feature, percentile)
    return percentiles


def np_spearman_correlation(feature1, feature2):
    return stats.spearmanr(feature1, feature2)


def data_types_to_df(data_frame):
    column_name_list = [column_name for column_name in data_frame.columns]
    column_type_list = [column_type.name for column_type in data_frame.dtypes]

    data_types = {'column_name': column_name_list, 'column_type': column_type_list}
    return pd.DataFrame.from_dict(data_types)


def main(data_frame):
    answer = dict()

    data_types = data_types_to_df(data_frame)
    answer['data_types'] = df_to_json(data_types)

    distributions = dict()

    for column_name in data_frame.columns:
        feature = data_frame[column_name]
        distributions[column_name] = ax_to_json(pd_get_distribution(feature))
        plt.close()

    answer['distributions'] = distributions

    correlation_table = data_frame.corr(method='spearman')
    answer['correlation_table'] = correlation_table

    answer['description'] = df_to_json(data_frame.describe())

    return answer


def main2(x, y):
    answer = dict()

    for feature_index in range(x.shape[1]):
        feature = x[:, feature_index]
        get_distribution(feature)

    get_distribution(y)

    answer['x_distribution'] = []
    answer['y_distribution'] = []

    correlation_table = np.zeros((x.shape[1], x.shape[1]))

    for feature1_index in range(x.shape[1]):
        feature1 = x[:, feature1_index]
        percentiles = get_percentiles(feature1, [0.5, 2.5, 5, 25, 50, 75, 95, 97.5, 99.5])

        for feature2_index in range(x.shape[1]):
            feature2 = x[:, feature2_index]
            coef, zeros_hypothesis = spearman_correlation(feature1, feature2)
            correlation_table[feature1_index, feature2_index] = coef

    answer['correlation_table'] = correlation_table.tolist()

    return answer
