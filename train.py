"""
Scrip to train a given model
"""
import pickle

from model.scikit_learn.model import ScikitLogisticReg
from model.diy.model import DiyLogisticReg
import tools


def train(model_type, path_pickle):
    """Main function."""
    # Init the model
    model_instance = init_model(model_type=model_type)
    # Load the data
    x_array, y_array = load_data(path_pickle)
    # Train the model
    trained_model = fit(model_instance, x_array, y_array)
    # Persist the model
    path_pickle = "model/{}/trained_model.pkl".format(model_type)
    persist_model(trained_model, path_pickle)


def init_model(model_type):
    """Init the model."""
    if model_type == "scikit_learn":
        model_instance = ScikitLogisticReg()
    elif model_type == "diy":
        model_instance = DiyLogisticReg()
    return model_instance


@tools.debug
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


@tools.timeit
def fit(model, x_array, y_array):
    """Train a given model"""
    model.fit(x_array, y_array)
    return model


def persist_model(trained_model, path_pickle):
    """Persist the model."""
    trained_model.persist(path_pickle)


if __name__ == '__main__':
    PATH_PICKLE = "data/kaggle/data_array.pkl"
    MODEL_TYPE = "scikit_learn"
    train(MODEL_TYPE, PATH_PICKLE)
