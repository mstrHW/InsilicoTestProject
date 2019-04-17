from module.data_loader.mongodb_loader import MongoDBLoader
from module.models.model_factory import make_model
import logging
import matplotlib.pyplot as plt
from definitions import *


def print_graphs(mongodb_loader: MongoDBLoader, columns: List[str]):
    for key in columns:
        logging.info(key + ' graph')
        mongodb_loader.load_graph('metrics', key)
        plt.show()


def print_statistics_graphs(mongodb_loader: MongoDBLoader, columns: List[str]):
    for key in columns:
        label = '{}_distribution'.format(key)
        logging.info(label + ' graph')
        mongodb_loader.load_graph('statistics', label)
        plt.show()


def main(model_name: str, mongo_dataset_name: str):
    mongodb_loader = MongoDBLoader(mongo_dataset_name)
    train_data, test_data = mongodb_loader.load_train_test('dataset')

    (x_train, y_train) = train_data
    (x_test, y_test) = test_data
    columns = x_train.columns.tolist() + y_train.columns.tolist()

    logging.info('x_train data:')
    logging.info(x_train.head(10))

    logging.info('y_train data:')
    logging.info(y_train.head(10))

    data_types, correlation_table, description = mongodb_loader.load_statistics(columns)
    logging.info('data_types:')
    logging.info(data_types)
    logging.info('correlation_table:')
    logging.info(correlation_table)
    logging.info('description:')
    logging.info(description)

    print_statistics_graphs(mongodb_loader, columns)
    print_graphs(mongodb_loader, ['heatmap', 'roc'])

    model = make_model(model_name)
    model = mongodb_loader.load_model(model.name)
    logging.info('model:')
    logging.info(model)

    y_pred = model.classifier.predict(x_test)   # TODO: saving and loading
    y_probabilities = model.classifier.predict_proba(x_test)

    metrics_report = mongodb_loader.load_metrics()
    logging.info('metrics_report:')
    logging.info(metrics_report)


if __name__ == '__main__':
    main()
