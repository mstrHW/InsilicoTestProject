from sklearn.tree import DecisionTreeClassifier


class DTModel(object):

    name = 'sklearn_decision_tree'

    def __init__(self):
        self.classifier = DecisionTreeClassifier()

    def train(self, x, y):
        self.classifier.fit(x, y)

    def test(self, x, y):
        return self.classifier.score(x, y)

    def predict(self, x):
        return self.classifier.predict(x)

    def predict_proba(self, x):
        return self.classifier.predict_proba(x)

    def get_grid_search_parameters(self):
        parameters = {'min_samples_split': range(10, 500, 20), 'max_depth': range(1, 20, 2)}
        return parameters

    def set_params(self, best_params_):
        self.classifier.set_params(**best_params_)
