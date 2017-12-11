"""
Gather the utils function.
"""
import pickle


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
