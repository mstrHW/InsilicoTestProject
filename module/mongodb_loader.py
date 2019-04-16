import pymongo
import json
import numpy as np
import pandas as pd
import pickle

from definitions import *


def dict_to_json(dict_: Dict) -> Dict:
    return json.dumps(dict_)


def np_to_json(array: np.array) -> Dict:
    return json.dumps(array.tolist())


def df_to_json(data_frame: pd.DataFrame, orient='columns') -> Dict:
    return data_frame.to_json(orient=orient) # orient='records'


def ax_to_json(axes_subplot) -> Dict:
    return pickle.dumps(axes_subplot)


class MongoDBLoader(object):
    def __init__(self, database_name: str, drop_existing: bool = False):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.database_name = database_name
        if drop_existing:
            self.client.drop_database(database_name)
        self.database = self.client[database_name]

    def insert_data(self, table_name, label, json_data):
        self.database[table_name].insert_one({'label': label, 'data': json_data})

    def find_data(self, table_name, label):
        json_answer = self.database[table_name].find_one({'label': label})['data']
        pd_df = pd.read_json(json_answer)
        # pd.read_json(json_answer).values
        return pd_df

    def save_graph(self, table_name, label, graph):
        # graph = pickle.dumps(graph)
        self.database[table_name].insert_one({'type': 'graph', 'label': label, 'graph': graph})

    def save_correlation_table(self, table_name, label, correlation_table):
        json_data = df_to_json(correlation_table, 'values')
        self.database[table_name].insert_one({'type': 'correlation_table', 'label': label, 'data': json_data})

    def load_correlation_table(self, table_name, label):
        json_data = self.database[table_name].find_one({'type': 'correlation_table', 'label': label})['data']
        return pd.read_json(json_data, 'values')

    def load_graph(self, table_name, label):
        pickled_graph = self.database[table_name].find_one({'type': 'graph', 'label': label})['graph']
        return pickle.loads(pickled_graph)

    def load_train_test(self, table_name: str):
        x_train = self.find_data(table_name, 'x_train')
        y_train = self.find_data(table_name, 'y_train')

        x_test = self.find_data(table_name, 'x_test')
        y_test = self.find_data(table_name, 'y_test')

        return (x_train, y_train), (x_test, y_test)

    def save_model(self, model_name: str, model: object):
        self.database['models'].remove({'model_name': model_name})
        pickled_model = pickle.dumps(model)
        self.database['models'].insert_one({'model_name': model_name, 'model': pickled_model})

    def load_model(self, model_name: str):
        pickled_model = self.database['models'].find_one({'model_name': model_name})['model']
        return pickle.loads(pickled_model)

    def save_predictions(self, model_name: str, predictions):
        self.database['models'].update({'model_name': model_name}, {'predictions': predictions})

