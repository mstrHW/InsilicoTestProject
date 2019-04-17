from module.mongodb_loader import MongoDBLoader, np_to_json
from module.models.model_factory import make_nn_model
from module.metrics import get_f1_per_class, multiclass_roc_auc_score, confusion_matrix
from module.plot_graphs import plot_confusion_matrix, plot_roc_per_class

from sklearn.model_selection import cross_val_predict
import numpy as np


def cross_val(model, train_data, test_data, mongodb_loader):
    (x_train, y_train) = train_data
    (x_test, y_test) = test_data

    train_predictions = cross_val_predict(model.classifier, x_train, y_train, cv=5, method='predict_proba')
    test_predictions = cross_val_predict(model.classifier, x_test, y_test, cv=5, method='predict_proba')

    mongodb_loader.insert_data('predictions', 'train_cross_val_predictions', np_to_json(train_predictions))
    mongodb_loader.insert_data('predictions', 'test_cross_val_predictions', np_to_json(test_predictions))


def calculate_metrics(y_test, y_pred, y_probabilities):
    metrics_report = get_f1_per_class(y_test, y_pred)
    auc = multiclass_roc_auc_score(y_test, y_probabilities)

    for class_index in range(y_probabilities.shape[1]):
        metrics_report[str(class_index)]['auc'] = auc[class_index]

    print(metrics_report)
    print(auc)

    return metrics_report


def save_metrics(metrics, mongodb_loader):
    mongodb_loader.insert_data('metrics', 'metrics', metrics)


def main():
    mongodb_loader = MongoDBLoader('iris_dataset2')
    train_data, test_data = mongodb_loader.load_train_test('dataset')

    (x_train, y_train) = train_data
    (x_test, y_test) = test_data

    model = make_nn_model()

    cross_val(model, train_data, test_data, mongodb_loader)

    parameters = {'solver': ['lbfgs'], 'max_iter': [1000, 1500, 2000],
                  'alpha': 10.0 ** -np.arange(1, 10, 3), 'hidden_layer_sizes': np.arange(10, 100, 10),
                  'random_state': [0, 3, 6, 9]}
    from sklearn.model_selection import GridSearchCV

    clf = GridSearchCV(model.classifier, parameters, n_jobs=-1)

    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))
    print(clf.best_params_)

    model.classifier.set_params(**clf.best_params_)
    model.classifier.fit(x_train, y_train)

    print(model.classifier.score(x_train, y_train))
    print(model.classifier.score(x_test, y_test))
    model_name = model.name

    mongodb_loader.save_model(model_name, model)

    y_pred = model.classifier.predict(x_test)
    y_probabilities = model.classifier.predict_proba(x_test)

    metrics_report = calculate_metrics(y_test, y_pred, y_probabilities)
    save_metrics(metrics_report, mongodb_loader)

    matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(matrix)
    plot_roc_per_class(y_test, y_probabilities)


if __name__ == '__main__':
    main()
