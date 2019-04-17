import numpy as np

from module.data_loader.data_preprocessing import normalize_data


def test_normalization_realizations():
    x = [
        [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9],
        [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1],
        [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5],
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1],
    ]
    x = np.array(x)
    x = np.moveaxis(x, 0, -1)

    def handwritten_normalization(x: np.array):
        return (x - x.min()) / (x.max() - x.min())

    hw_normalized_x = np.zeros(x.shape)
    for feature_index in range(hw_normalized_x.shape[1]):
        hw_normalized_x[:, feature_index] = handwritten_normalization(x[:, feature_index])

    normalized_x = normalize_data(x)

    assert(np.allclose(normalized_x, hw_normalized_x, atol=1e-15))
