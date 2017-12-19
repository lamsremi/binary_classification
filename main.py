"""
script to use the logistic regression model
"""
import pickle

from model.diy.model import DiyLogisticReg
# import tools


# @tools.debug
def predict(x_array, model_string, model=None):
    """Main function."""
    if model is None:
        # Load the model
        model = load_model(model_string=model_string)
    # Predict
    y_array = model.predict(x_array)
    return y_array, model


def load_model(model_string):
    """Load a particular model."""
    if model_string == "scikit_learn":
        path_sav = "model/{}/trained_model.sav".format(model_string)
        loaded_model = pickle.load(open(path_sav, 'rb'))
    if model_string == "diy":
        loaded_model = DiyLogisticReg()
    return loaded_model


# if __name__ == '__main__':
#     MODEL_STRING = "scikit_learn"
#     predict(MODEL_STRING)
