from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

from module import calculate_statistics
from module.load_sklearn_data import load_data
from module.data_preprocessing import normalize_data
from module.mongodb_loader import *
from module.nn_model import NNModel
from module.plot_graphs import plot_confusion_matrix, plot_roc_per_class
from module.metrics import multiclass_roc_auc_score, get_f1_per_class


def first_part():
    test_size = 0.33
    x, y = load_data()
    statistics = calculate_statistics.main(x, y)
    x = normalize_data(x)
    # y = binarize_target(y)

    print(statistics)
    print(x)

    mydb = get_db()
    data_table = mydb['data']
    insert_data(data_table, 'normalized_x', np_to_json(x))
    insert_data(data_table, 'statistics', json.dumps(statistics))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    insert_data(data_table, 'train_x', json.dumps(X_train.tolist()))
    insert_data(data_table, 'train_y', json.dumps(y_train.tolist()))

    insert_data(data_table, 'test_x', json.dumps(X_test.tolist()))
    insert_data(data_table, 'test_y', json.dumps(y_test.tolist()))


def cross_val(model, mydb, train_x, train_y, test_x, test_y):
    _train_y = [np.argmax(y) for y in train_y]
    _test_y = [np.argmax(y) for y in test_y]
    train_predictions = cross_val_predict(model.classifier, train_x, _train_y, cv=5, method='predict_proba')
    test_predictions = cross_val_predict(model.classifier, test_x, _test_y, cv=5, method='predict_proba')

    predictions_table = mydb['predictions']
    insert_data(predictions_table, 'train_cross_val_predictions', np_to_json(train_predictions))
    insert_data(predictions_table, 'test_cross_val_predictions', np_to_json(test_predictions))


def second_part():
    mydb = get_db()
    data_table = mydb['data']

    train_x_json = find_data(data_table, 'train_x')['data']
    train_x = pd.read_json(train_x_json).values

    train_y_json = find_data(data_table, 'train_y')['data']
    train_y = pd.read_json(train_y_json).values

    test_x_json = find_data(data_table, 'test_x')['data']
    test_x = pd.read_json(test_x_json).values

    test_y_json = find_data(data_table, 'test_y')['data']
    test_y = pd.read_json(test_y_json).values

    print(train_x)
    print(train_y)

    model = NNModel()
    model.train(train_x, train_y)
    model_name = model.name

    model_table = mydb['model']
    pickled_model = pickle.dumps(model)
    insert_data(model_table, model_name, pickled_model)

    print(model.test(train_x, train_y))
    print(model.test(test_x, test_y))

    y_pred = model.classifier.predict(test_x)
    y_pred_proba = model.classifier.predict_proba(test_x)

    metrics_report = get_f1_per_class(test_y, y_pred)
    auc = multiclass_roc_auc_score(test_y, y_pred_proba)

    for class_index in ['0', '1', '2']:
        metrics_report[class_index]['auc'] = auc[int(class_index)]

    # _test_y = [np.argmax(y) for y in test_y]
    # _y_pred = [np.argmax(y) for y in y_pred]
    matrix = confusion_matrix(test_y, y_pred)
    plot_confusion_matrix(matrix)

    plot_roc_per_class(test_y, y_pred_proba)
    print(metrics_report)
    print(auc)


def main():
    # first_part()
    second_part()


if __name__ == '__main__':
    main()
