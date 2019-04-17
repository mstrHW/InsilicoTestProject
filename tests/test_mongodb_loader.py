import numpy as np

from module.mongodb_loader import MongoDBLoader
from module.models.nn_model import NNModel

# def test_cr():
#     x = [
#         [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9],
#         [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1],
#         [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5],
#         [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1],
#     ]
#     x = np.array(x)
#     x = np.moveaxis(x, 0, -1)
#
#     def handwritten_normalization(x: np.array):
#         return (x - x.min()) / (x.max() - x.min())
#
#     hw_normalized_x = np.zeros(x.shape)
#     for feature_index in range(hw_normalized_x.shape[1]):
#         hw_normalized_x[:, feature_index] = handwritten_normalization(x[:, feature_index])
#
#     normalized_x = normalize_data(x)
#
#     assert(np.allclose(normalized_x, hw_normalized_x, atol=1e-15))


def test_pickle_model():
    mongodb_loader = MongoDBLoader('iris_dataset')
    train_data, test_data = mongodb_loader.load_train_test('data')

    (x_train, y_train) = train_data
    (x_test, y_test) = test_data

    model = NNModel()
    model.train(x_train, y_train)
    model_name = model.name

    train_accuracy = model.test(x_train, y_train)
    test_accuracy = model.test(x_test, y_test)

    mongodb_loader.save_model(model_name, model)
    model = mongodb_loader.load_model(model_name)

    loaded_train_accuracy = model.test(x_train, y_train)
    loaded_test_accuracy = model.test(x_test, y_test)

    assert(np.allclose(train_accuracy, loaded_train_accuracy))
    assert(np.allclose(test_accuracy, loaded_test_accuracy))
