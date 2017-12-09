"""
script to use the logistic regression model
"""
import pickle

import tools


@tools.debug
def predict(model_type, x_array):
    """Main function."""
    # Load the model
    loaded_model = load_model(model_type=model_type)
    # Predict
    y_array = loaded_model.predict(x_array)
    return y_array


def load_model(model_type):
    """Load a particular model."""
    path_pickle = "model/{}/trained_model.pkl".format(model_type)
    loaded_model = pickle.loads(path_pickle)
    return loaded_model


if __name__ == '__main__':
    predict()
