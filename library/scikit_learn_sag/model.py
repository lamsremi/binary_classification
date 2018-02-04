"""
Scikit_learn based implementation of logistic regression.

Description of the module.
"""
import sys
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class Model():
    """Logistic Regression Class.
    """
    def __init__(self):
        """Init class.
        """
        self._model = None

    def predict(self, data_df):
        """Prediction function of the model.
        Arg:
            data_df (DataFrame): table of input data to predict.
        Return
            prediction_df (DataFrame): table of prediction.
        """
        # Set prediction table
        prediction_df = pd.DataFrame()

        # Iterate
        for row in data_df.itertuples():
            prediction_df.loc[row[0], "prediction"] = self.predict_record(row)

        # Return the prediction table
        return prediction_df

    def predict_record(self, row):
        """Predict one record.
        Args:
            row (Pandas Object Row): input of one record.
        Return:
            prediction (float): prediction.
        """
        # Format and reshape for single sample
        x_array = np.array([value for value in row[1:]]).reshape(1, -1)
        # Compute
        y_array = self._model.predict(x_array)
        # Return prediction
        return y_array

    def fit(self, data_df, alpha, epochs):
        """Fit the model.
        Args:
            data_df (DataFrame): labeled dataset for fitting.
        Note:
            * Scikit-learn framework:
                Input: array-like, shape = [n_samples, n_features]
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
                Output: array-like
                y_array = [
                    y_1,
                    y_2
                ]
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

        x_array, y_array = self.format_data(data_df)
        self._model.fit(x_array, y_array)

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


    @staticmethod
    # @tools.debug
    def format_data(data_df):
        """Transform the data in the proper format.

        So it can be used by the method predict of scikit learn.

        Args:
            data_df (DataFrame): training dataset.
        """
        x_array = np.array(data_df.iloc[:, :-1])
        y_array = np.array(data_df.iloc[:, -1])
        return x_array, y_array
