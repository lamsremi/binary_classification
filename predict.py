"""
script to use the logistic regression model
"""
import pickle

from model.diy.model import DiyLogisticReg

# import tools

# @tools.debug
def main(x_array, model_type, model=None):
    """Main function."""
    if model is None:
        # Load the model
        model = load_model(model_type)
    # Predict
    y_array = model.predict(x_array)
    return y_array, model


def load_model(model_type):
    """Load a particular model."""
    if model_type == "scikit_learn":
        path_sav = "model/{}/trained_model.sav".format(model_type)
        loaded_model = pickle.load(open(path_sav, 'rb'))
    elif model_type == "diy":
        path_pickle = "model/{}/trained_model.pkl".format(model_type)
        loaded_model = DiyLogisticReg()
        loaded_model.load(path_pickle)
    return loaded_model


if __name__ == '__main__':
    X_ARRAY = None
    MODEL_TYPE = "diy"
    main(X_ARRAY, MODEL_TYPE)
