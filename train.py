"""
Scrip to train a given model
"""
from model.logisticRegression.scikit_learn.model import ScikitLogisticReg
from model.logisticRegression.diy.model import DiyLogisticReg

import tools
import utils


def train(model_type, model_source, path_pickle):
    """Main function."""
    # Load the data
    x_array, y_array = utils.load_data(path_pickle)
    # Filter data
    x_array, y_array = utils.filter_array(x_array, y_array, 0, 600)
    # Normalize data
    x_array = utils.normalize(x_array)
    # Init the model
    model_instance = init_model(model_type=model_type, model_source=model_source)
    # Train the model
    trained_model = fit(model_instance, x_array, y_array)
    # # Persist the model
    # persist_model(trained_model, path_pickle="model/{}/trained_model.sav".format(model_type))


def init_model(model_type, model_source):
    """Init the model."""
    if model_type == "logisticRegression":
        if model_source == "scikit_learn":
            model_instance = ScikitLogisticReg()
        elif model_source == "diy":
            model_instance = DiyLogisticReg()
    return model_instance


@tools.timeit
def fit(model, x_array, y_array):
    """Train a given model"""
    model.fit(x_array, y_array)
    return model


def persist_model(trained_model, path_pickle):
    """Persist the model."""
    trained_model.persist(path_pickle)


if __name__ == '__main__':
    PATH_PICKLE = "data/diabete/data_array.pkl"
    MODEL_TYPE = "logisticRegression"
    MODEL_SOURCE = "diy"
    train(MODEL_TYPE, MODEL_SOURCE, PATH_PICKLE)
