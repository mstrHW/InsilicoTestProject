from sklearn.neural_network import MLPClassifier


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
