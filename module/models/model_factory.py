from module.models.nn_model import NNModel
from module.models.dt_model import DTModel
from definitions import *


def choose_model_dict() -> Dict:
    choose_model = {
        'neural_network': make_nn_model,
        'decision_tree': make_decision_tree_model,
    }
    return choose_model


def possible_models() -> List[str]:
    return list(choose_model_dict().keys())


def make_nn_model() -> NNModel:
    return NNModel()


def make_decision_tree_model() -> DTModel:
    return DTModel()


def make_model(model_name: str):
    choose_model = choose_model_dict()
    return choose_model[model_name]()
