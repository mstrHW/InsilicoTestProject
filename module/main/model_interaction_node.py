from module.data_loader.mongodb_loader import MongoDBLoader, np_to_json
from module.models.model_factory import make_model
from module.utils.metrics import get_f1_per_class, multiclass_roc_auc_score, confusion_matrix
from module.utils.plot_graphs import plot_confusion_matrix, plot_roc_per_class

from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


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

    return metrics_report


def get_best_parameters(model, x_train, y_train):
    parameters = model.get_grid_search_parameters()
    clf = GridSearchCV(model.classifier, parameters, n_jobs=-1)
    clf.fit(x_train, y_train)
    return clf.best_params_


def main(model_name, mongo_dataset_name):
    mongodb_loader = MongoDBLoader(mongo_dataset_name)
    train_data, test_data = mongodb_loader.load_train_test('dataset')

    (x_train, y_train) = train_data
    (x_test, y_test) = test_data

    model = make_model(model_name)

    cross_val(model, train_data, test_data, mongodb_loader)

    best_params = get_best_parameters(model, x_train, y_train)

    model.set_params(best_params)
    model.train(x_train, y_train)

    mongodb_loader.save_model(model.name, model)

    y_pred = model.classifier.predict(x_test)
    y_probabilities = model.classifier.predict_proba(x_test)

    metrics_report = calculate_metrics(y_test, y_pred, y_probabilities)
    mongodb_loader.save_metrics(metrics_report)

    matrix = confusion_matrix(y_test, y_pred)

    heatmap = plot_confusion_matrix(matrix)
    plt.close()
    roc = plot_roc_per_class(y_test, y_probabilities)
    plt.close()

    mongodb_loader.save_graphs(heatmap, roc)


if __name__ == '__main__':
    main()
