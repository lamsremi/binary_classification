"""Prediction script.
"""
import sys
import importlib

import tools


@tools.debug
def main(inputs_data,
         model_type,
         model_version):
    """Perform a prediction.
    Args:
        inputs_data (list): dataset to predict.
        model_type (str): type of model to use for prediction.
        model_version (str): version of this type of model to use.
    Return
        (list): predictions.
    """
    # Init the model
    model = init_model(model_type)

    # Load the model
    model.load_parameters(model_version)

    # Perform the prediction
    return model.predict(inputs_data)


def init_model(model_type):
    """Instanciate an instance of the class of the given type of model.
    Args:
        model_type (str): type of model to train.
    Return:
        model (Model): instance of Model.
    """
    # Import model
    model_module = importlib.import_module("library.{}.model".format(model_type))
    # Return the instance
    return model_module.Model()


if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else "pure_python"
    inputs_data = [
        [18.0, 7.0, 4.0, 2.0, 6.0, 3.0, 61.0, 7.0, 24.0],
        [0.0, 7.0, 6.0, 2.0, 6.0, 6.0, 41.0, 4.0, 24.0]
    ]
    main(inputs_data,
         model_type=model,
         model_version="X")
