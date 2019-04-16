from sklearn.model_selection import train_test_split

from module.mongodb_loader import MongoDBLoader, np_to_json, dict_to_json, df_to_json
from load_sklearn_data import load_data
import calculate_statistics
from data_preprocessing import df_normalize_data
import matplotlib.pyplot as plt


def np_insert_splitted_dataset(x, y, mongodb_loader, data_table, test_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    mongodb_loader.insert_data(data_table, 'x_train', np_to_json(x_train))
    mongodb_loader.insert_data(data_table, 'y_train', np_to_json(y_train))

    mongodb_loader.insert_data(data_table, 'x_test', np_to_json(x_test))
    mongodb_loader.insert_data(data_table, 'y_test', np_to_json(y_test))


def insert_splitted_dataset(data_frame, mongodb_loader, table_name, test_size):
    feature_names = data_frame.columns
    target = 'target'
    feature_names = feature_names.drop(target)

    train, test = train_test_split(data_frame, test_size=test_size)
    mongodb_loader.insert_data(table_name, 'x_train', df_to_json(train[feature_names], 'values'))
    mongodb_loader.insert_data(table_name, 'y_train', df_to_json(train[target], 'values'))

    mongodb_loader.insert_data(table_name, 'x_test', df_to_json(test[feature_names], 'values'))
    mongodb_loader.insert_data(table_name, 'y_test', df_to_json(test[target], 'values'))


def main(test_size=0.33):
    data_frame, feature_names = load_data()
    mongodb_loader = MongoDBLoader('iris_dataset2', drop_existing=True)
    statistics = calculate_statistics.main(data_frame)
    data_frame[feature_names] = df_normalize_data(data_frame[feature_names])
    # y = binarize_target(y)

    mongodb_loader.insert_data('data', 'preprocessed_x', df_to_json(data_frame))

    data_types = statistics['data_types']
    mongodb_loader.insert_data('statistics', 'data_types', data_types)

    data_types = mongodb_loader.find_data('statistics', 'data_types')
    print(data_types)

    distributions = statistics['distributions']
    for key in distributions:
        mongodb_loader.save_graph('statistics', '{}_distribution'.format(key), distributions[key])

    for key in statistics['distributions']:
        mongodb_loader.load_graph('statistics', '{}_distribution'.format(key))
        # plt.show()

    correlation_table = statistics['correlation_table']
    mongodb_loader.save_correlation_table('statistics', 'correlation_table', correlation_table)

    correlation_table = mongodb_loader.load_correlation_table('statistics', 'correlation_table')
    correlation_table.columns = feature_names + ['target']
    correlation_table.index = feature_names + ['target']
    print(correlation_table)

    description = statistics['description']
    mongodb_loader.insert_data('statistics', 'description', description)

    description = mongodb_loader.find_data('statistics', 'description')
    print(description)

    # for key in statistics.keys():
    #     pd_df = mongodb_loader.find_data('statistics', key)
    #     print(pd_df)

    insert_splitted_dataset(data_frame, mongodb_loader, 'dataset', test_size)


if __name__ == '__main__':
    main(0.33)
