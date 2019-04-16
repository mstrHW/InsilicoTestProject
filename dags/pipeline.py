from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

import logging
from datetime import datetime

from load_sklearn_data import load_data
import calculate_statistics
# from module.mongodb_loader import


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2019, 2, 16),
}


with DAG('full_pipeline', default_args=default_args) as dag:
    task_load_data_id = 'load_data'
    task_load_data = PythonOperator(
        task_id=task_load_data_id,
        provide_context=True,
        op_kwargs={
            'image_name': task_load_data_id,
            'my_id': task_load_data_id
        },
        python_callable=load_data
    )

    task_calculate_statistics_id = 'calculate_statistics'
    task_calculate_statistics = PythonOperator(
        task_id=task_calculate_statistics_id,
        provide_context=True,
        op_kwargs={
            'image_name': task_calculate_statistics_id,
            'my_id': task_calculate_statistics_id
        },
        python_callable=calculate_statistics.main
    )

    task_save_statistics_id = 'save_statistics'
    task_save_statistics = PythonOperator(
        task_id=task_save_statistics_id,
        provide_context=True,
        op_kwargs={
            'image_name': task_save_statistics_id,
            'my_id': task_save_statistics_id
        },
        # python_callable=calculate_statistics.main
    )

    task_data_preprocessing_id = 'data_preprocessing'
    task_data_preprocessing = PythonOperator(
        task_id=task_data_preprocessing_id,
        provide_context=True,
        op_kwargs={
            'image_name': task_data_preprocessing_id,
            'my_id': task_data_preprocessing_id
        },
        # python_callable=calculate_statistics.main
    )

    task_train_test_split_id = 'train_test_split'
    task_train_test_split = PythonOperator(
        task_id=task_train_test_split_id,
        provide_context=True,
        op_kwargs={
            'image_name': task_train_test_split_id,
            'my_id': task_train_test_split_id
        },
        # python_callable=calculate_statistics.main
    )

    task_save_data_id = 'save_data'
    task_save_data = PythonOperator(
        task_id=task_save_data_id,
        provide_context=True,
        op_kwargs={
            'image_name': task_save_data_id,
            'my_id': task_save_data_id
        },
        # python_callable=calculate_statistics.main
    )

    task_load_data >> [task_calculate_statistics, task_data_preprocessing]
    task_data_preprocessing >> task_train_test_split >> task_save_data


