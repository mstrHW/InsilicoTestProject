import pickle

from module.mongodb_loader import MongoDBLoader
from module.nn_model import NNModel
from module.metrics import get_f1_per_class, multiclass_roc_auc_score, confusion_matrix
from module.plot_graphs import plot_confusion_matrix, plot_roc_per_class


def cross_val(model, mydb, train_x, train_y, test_x, test_y):
    _train_y = [np.argmax(y) for y in train_y]
    _test_y = [np.argmax(y) for y in test_y]
    train_predictions = cross_val_predict(model.classifier, train_x, _train_y, cv=5, method='predict_proba')
    test_predictions = cross_val_predict(model.classifier, test_x, _test_y, cv=5, method='predict_proba')

    predictions_table = mydb['predictions']
    insert_data(predictions_table, 'train_cross_val_predictions', np_to_json(train_predictions))
    insert_data(predictions_table, 'test_cross_val_predictions', np_to_json(test_predictions))


def main():
    mongodb_loader = MongoDBLoader('iris_dataset2')
    train_data, test_data = mongodb_loader.load_train_test('dataset')

    (x_train, y_train) = train_data
    (x_test, y_test) = test_data

    model = NNModel()
    model.train(x_train, y_train)
    model_name = model.name

    mongodb_loader.save_model(model_name, model)

    y_pred = model.classifier.predict(x_test)
    y_probabilities = model.classifier.predict_proba(x_test)

    metrics_report = get_f1_per_class(y_test, y_pred)
    auc = multiclass_roc_auc_score(y_test, y_probabilities)

    for class_index in range(y_probabilities.shape[1]):
        metrics_report[str(class_index)]['auc'] = auc[class_index]

    matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(matrix)

    plot_roc_per_class(y_test, y_probabilities)
    print(metrics_report)
    print(auc)


if __name__ == '__main__':
    main()
