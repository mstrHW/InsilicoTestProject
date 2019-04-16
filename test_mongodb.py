from sklearn import datasets
import json

from module.mongodb_loader import *

data = datasets.load_iris(return_X_y=False)

mydb = get_db()
data_db = mydb['data']

# json_data = json.dumps(data.data.tolist())
# json_target = json.dumps(data.target.tolist())
#
# data_db.insert({'name': 'x', 'data': json_data})
# data_db.insert({'name': 'y', 'data': json_target})

json_data = find_data(data_db, 'x')['data']
print(json_data)
json_target = find_data(data_db, 'y')['data']
print(json_target)
