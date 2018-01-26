"""
Gather the utils function.
"""
import pickle
import numpy as np

import tools

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
