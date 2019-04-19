# InsilicoTestProject
main.py for all use cases.

parameters:

--process_type: one of (all_process, run_modules or print_results)

--test_size: test_size for train_test_split (0..1), type=float

--model_name: one of implemented models (NNModel or DecisionTreeModel)

--dataset_name: one of available datasets (iris)

--mongo_dataset_name: what name to save dataset in mongo (equals to the dataset_name by default)
If a dataset with the same name is found in MongoDB, it will be deleted.

--logging_file: (IN THE NEXT VERSION) logging_file (if None, the logging information will be displayed in the console)
    
    
airflow_home/dags - does not work
