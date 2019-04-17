from module.main import data_processing_node, model_interaction_node, print_saved_data
from module.models.model_factory import possible_models
from module.data_loader.load_sklearn_data import possible_datasets
import argparse


def main(args):
    test_size = 0.33
    model_name = 'neural_network'
    process_type = 'all_process'
    dataset_name = 'iris'
    mongo_dataset_name = dataset_name
    logging_file = args.logging_file

    if args.test_size:
        test_size = args.test_size
    if args.model_name:
        if args.model_name in possible_models():
            model_name = args.model_name
        else:
            # TODO: error
            pass
    if args.process_type:
        process_type = args.process_type
    if args.dataset_name:
        if args.dataset_name in possible_datasets():
            dataset_name = args.dataset_name
            mongo_dataset_name = dataset_name
        else:
            # TODO: error
            pass
    if args.mongo_dataset_name:
        mongo_dataset_name = dataset_name

    base_modules = False
    print_module = False
    if process_type == 'all_process':
        base_modules = True
        print_module = True
    elif process_type == 'run_modules':
        base_modules = True
    elif process_type == 'print_results':
        print_module = True
    else:
        # TODO: error
        pass

    if base_modules:
        data_processing_node.main(test_size, dataset_name, mongo_dataset_name)
        model_interaction_node.main(model_name, mongo_dataset_name)
    if print_module:
        print_saved_data.main(model_name, mongo_dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_type', help='one of (all_process, run_modules or print_results)')
    parser.add_argument('--test_size', help='test_size for train_test_split (0..1)', type=float)
    parser.add_argument('--model_name', help='one of implemented models (NNModel or DecisionTreeModel)')
    parser.add_argument('--dataset_name', help='one of available datasets (iris)')
    parser.add_argument('--mongo_dataset_name', help='what name to save dataset in mongo (equals to the dataset_name by default)')
    parser.add_argument('--logging_file', help='(IN THE NEXT VERSION) logging_file (if None, the logging information will be displayed in the console')
    main(parser.parse_args())
