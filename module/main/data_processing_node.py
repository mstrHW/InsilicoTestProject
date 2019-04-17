from module.data_loader.mongodb_loader import MongoDBLoader, df_to_json
from module.data_loader.load_sklearn_data import load_dataset
from module.utils.calculate_statistics import calculate_statistics
from module.data_loader.data_preprocessing import normalize_data, split_dataset


def main(test_size: float, dataset_name: str, mongo_dataset_name: str):
    mongodb_loader = MongoDBLoader(mongo_dataset_name, drop_existing=True)

    data_frame, feature_names, target_name = load_dataset(dataset_name)

    statistics = calculate_statistics(data_frame)
    mongodb_loader.save_statistics(statistics)

    data_frame[feature_names] = normalize_data(data_frame[feature_names])
    mongodb_loader.insert_data('data', 'preprocessed_x', df_to_json(data_frame))

    train, test = split_dataset(data_frame, test_size)
    mongodb_loader.save_splitted_dataset(train, test, test_size, feature_names, target_name)

    return train, test

