from sklearn.neural_network import MLPClassifier
import numpy as np


class NNModel(object):

    name = 'sklearn_mlp_classifier'

    def __init__(self):
        self.classifier = MLPClassifier(hidden_layer_sizes=(25,))

    def train(self, x, y):
        self.classifier.fit(x, y)

    def test(self, x, y):
        return self.classifier.score(x, y)

    def predict(self, x):
        return self.classifier.predict(x)

    def predict_proba(self, x):
        return self.classifier.predict_proba(x)

    def get_grid_search_parameters(self):
        parameters = {'solver': ['lbfgs'], 'max_iter': [1000, 1500, 2000],
                      'alpha': 10.0 ** -np.arange(1, 10, 3), 'hidden_layer_sizes': np.arange(10, 100, 10),
                      'random_state': [0, 3, 6, 9]}
        return parameters

    def set_params(self, best_params_):
        self.classifier.set_params(**best_params_)
