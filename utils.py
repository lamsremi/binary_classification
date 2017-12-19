"""
Gather the utils function.
"""
import pickle
import numpy as np

# @tools.debug
def load_data(path_pickle):
    """
    Load the data fron pickle that is already in this format:
    x_array = [
        [
            X_11,
            X_12
        ],
        [
            X_21,
            X_22
        ]
    ]
    y_array = [
        y_1,
        y_2
    ]
    """
    # Load data
    data_array = pickle.load(open(path_pickle, "rb"))
    # Input
    x_array = [data[:-1] for data in data_array]
    # Output
    y_array = [data[-1] for data in data_array]
    return x_array, y_array


def filter_array(x_array, y_array, start_index, end_index):
    """
    Filter the data.
    """
    x_array = x_array[start_index:end_index]
    y_array = y_array[start_index:end_index]
    return x_array, y_array

def normalize(x_array):
    """Normalize data."""
    x_array = np.array(x_array)
    for j_index in range(x_array.shape[1]):
        mean = np.std(x_array[:, j_index])
        std = np.std(x_array[:, j_index])
        x_array[:, j_index] -= mean
        x_array[:, j_index] /= std
    return x_array
