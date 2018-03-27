"""Scikit_learn based implementation of logistic regression.
"""
import sys
import os
import pickle

from sklearn.linear_model import LogisticRegression


class Model():
    """Logistic Regression Class.
    """
    def __init__(self):
        """Instanciate object."""
        self._model = None

    def predict(self, inputs_data):
        """Prediction function of the model.
        Arg:
            inputs_data (DataFrame): table of input data to predict.
        Return
            prediction_df (DataFrame): table of prediction.
        """
        # Return the prediction list
        # return [self._predict_instance(row) for row in inputs_data]
        return list(self._model.predict(inputs_data))


    def fit(self, labeled_data, alpha, epochs):
        """Fit the model.
        Args:
            labeled_data (list): labeled dataset for fitting.
            [
                ([ 0.0, 6.0, 6.0, 2.0], 1.0)
                ([ 8.0, 7.0, 4.0, 2.0], 1.0)
            ]
        Note:
            * Scikit-learn framework:
                Input: array-like, shape = [n_samples, n_features]
                x_array = [
                    [X_11, X_12],
                    [X_21, X_22]
                ]
                Output: array-like
                y_array = [y_1, y_2]
        """
        # If no verison is given
        if self._model is None:
            # Load initial parameters.
            self._model = LogisticRegression(penalty="l2",
                                             dual=False,
                                             tol=0.0001,
                                             C=1000000000000,
                                             fit_intercept=True,
                                             intercept_scaling=1,
                                             class_weight=None,
                                             random_state=None,
                                             solver="sag",
                                             max_iter=epochs,
                                             multi_class="ovr",
                                             verbose=1,
                                             warm_start=False,
                                             n_jobs=1)
        # Extract input and labels
        x_arrays = [instance[0] for instance in labeled_data]
        y_arrays = [instance[1] for instance in labeled_data]
        # Fit
        self._model.fit(x_arrays, y_arrays)

    def load_parameters(self, model_version):
        """Load the parameters of the model.
        Args:
            model_version (str): version of model to use.
        """
        # If a version model was given
        if model_version is not None:
            # Try to set the folder path.
            try:
                folder_path = "library/scikit_learn_sag/params/" + model_version
            # If error type, catch the exception.
            except TypeError:
                print("The given model_version might not be of type string")
            # Try to load the model.
            try:
                with open(folder_path + "/" + "model.sav", "rb") as handle:
                    self._model = pickle.load(handle)
            # If error, catch the exception and stop the program.
            except OSError:
                print("Can't find the model to load, please choose an existing version.")
                sys.exit()


    def persist_parameters(self, model_version):
        """Persist the parameters of the model.
        Args:
            model_version (str): version of model to use.
        """
        try:
            # Set the folder path
            folder_path = "library/scikit_learn_sag/params/" + model_version
        except TypeError:
            print("The given version of model might not be of type string")

        # Create folder if doesn't exist
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        # Store the model.
        with open(folder_path + "/" + "model.sav", "wb") as handle:
            pickle.dump(self._model, handle)
