from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

# from definitions import *
import logging
from datetime import datetime

import sys
import os
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

from module.main import data_processing_node
from module.main import model_interaction_node
# from module.mongodb_loader import


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2019, 2, 16),
}

with DAG('ez_pipeline', default_args=default_args) as dag:
    task_data_preprocessing_id = 'data_preprocessing'
    task_data_preprocessing = PythonOperator(
        task_id=task_data_preprocessing_id,
        provide_context=True,
        op_kwargs={
            'image_name': task_data_preprocessing_id,
            'my_id': task_data_preprocessing_id
        },
        python_callable=data_processing_node.main
    )

    task_sklearn_model_training_id = 'sklearn_model_training'
    task_sklearn_model_training = PythonOperator(
        task_id=task_sklearn_model_training_id,
        provide_context=True,
        op_kwargs={
            'image_name': task_sklearn_model_training_id,
            'my_id': task_sklearn_model_training_id
        },
        python_callable=model_interaction_node.main
    )

    task_data_preprocessing >> task_sklearn_model_training


