"""Module to fit a model given a training dataset.
"""
import importlib

import pandas as pd

import tools


def main(train_data=None,
         dataset=None,
         model_type=None,
         start_version=None,
         end_version=None):
    """Fit a model.
    Args:
        train_data (DataFrame): dataset use to train
        dataset (str): source to take the training dataset from.
        model_type (str): type of model to train.
        start_version (str): version of model to start the training from.
        end_version (str): version to store the parameters.
    """
    # Load data if None given
    if train_data is None:
        train_data = load_data(dataset)

    # Init the model
    model = init_model(model_type)

    # Load parameters
    model.load_parameters(model_version=start_version)

    # Fit the model
    model.fit(train_data,
              alpha=0.00001,
              epochs=20)

    # Persist the parameters
    model.persist_parameters(model_version=end_version)

    # Return bool
    return True


def load_data(dataset):
    """Load traiing data.
    Args:
        dataset (str): source to take the data from.
    Return:
        train_data (DataFrame): loaded table.
    """
    # Load data
    train_data = pd.read_csv("data/{}/data.csv".format(dataset),
                             nrows=400)

    # Return the table
    return train_data


def init_model(model_type):
    """Init an instance of the model
    Args:
        model_type (str): type of model to train.
    Return:
        model (instance Object): init model.
    """
    # Import model
    model_module = importlib.import_module("library.{}.model".format(model_type))

    # Init
    model = model_module.Model()

    # Return model
    return model


if __name__ == '__main__':
    for source in ["us_election"]:
        for model in ["scikit_learn_sag", "diy"]:
            main(train_data=None,
                 dataset=source,
                 model_type=model,
                 start_version=None,
                 end_version="X")
