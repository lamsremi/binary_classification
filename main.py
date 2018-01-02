"""
script to use the logistic regression model
"""
import pickle

from model.logisticRegression.diy.model import DiyLogisticReg
# from model.decisionTree.diy.model import DiyDecisionTree

# import tools

# @tools.debug
def predict(x_array, model_type, model_source, model=None):
    """Main function."""
    if model is None:
        # Load the model
        model = load_model(model_type, model_source)
    # Predict
    y_array = model.predict(x_array)
    return y_array, model


def load_model(model_type, model_source):
    """Load a particular model."""
    if model_type == "logisticRegression":
        if model_source == "scikit_learn":
            path_sav = "model/{}/{}/trained_model.sav".format(model_type, model_source)
            loaded_model = pickle.load(open(path_sav, 'rb'))
        elif model_source == "diy":
            path_pickle = "model/{}/{}/trained_model.pkl".format(model_type, model_source)
            loaded_model = DiyLogisticReg()
            loaded_model.load(path_pickle)
    elif model_type == "decisionTree":
        if model_source == "diy":
            loaded_model = DiyDecisionTree()
    return loaded_model


if __name__ == '__main__':
    X_ARRAY = None
    MODEL_TYPE = "decisionTree"
    MODEL_SOURCE = "diy"
    predict(X_ARRAY, MODEL_TYPE, MODEL_SOURCE)
