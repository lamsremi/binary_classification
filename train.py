"""
Scrip to train a given model
"""
from model.scikit_learn.model import ScikitLogisticReg
from model.diy.model import DiyLogisticReg

import tools
import utils


def train(model_type, input_source):
    """Main function."""
    # Load the data
    x_array, y_array = utils.load_data(
        "data/{}/data_array.pkl".format(input_source))
    # Filter data
    # x_array, y_array = utils.filter_array(x_array, y_array, 0, 600)
    # Normalize data
    x_array = utils.normalize(x_array)
    # Init the model
    model_instance = init_model(model_type=model_type)
    # Train the model
    trained_model = fit(model_instance, x_array, y_array)
    # # Persist the model
    persist_model(trained_model, model_type)


def init_model(model_type):
    """Init the model."""
    if model_type == "scikit_learn":
        model_instance = ScikitLogisticReg()
    elif model_type == "diy":
        model_instance = DiyLogisticReg()
    return model_instance


@tools.timeit
def fit(model, x_array, y_array):
    """Train a given model"""
    model.fit(x_array, y_array)
    return model


def persist_model(trained_model, model_type):
    """Persist the model."""
    if model_type == "scikit_learn":
        trained_model.persist(
            path_sav="model/{}/trained_model.sav".format(model_type))
    elif model_type == "diy":
        trained_model.persist(
            path_pickle="model/{}/trained_model.pkl".format(model_type))


if __name__ == '__main__':
    MODEL_TYPE = "diy"
    INPUT_SOURCE = "us_election"
    train(MODEL_TYPE, INPUT_SOURCE)
