"""
model do it yourself logistic regression
"""
import os
import pickle

import numpy as np
import pandas as pd

import tools
from library.diy.utils import cost_function, derivative_cost_function
from library.diy.utils import compute_product, compute_sigmoid

class Model():
    """Class DiyLogisticReg."""

    def __init__(self):
        """Init the model instance.
        Attributes:
            _input_dim
            _theta_array
            _biais
        """
        self._input_dim = None
        self._theta_array = None
        self._biais = None


    def predict(self, data_df):
        """Perfom a prediction.
        Args:
            data_df (DataFrame): list of records to predict.
        Return:
            prediction_df (DataFrame):
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
        # Format for single sample
        x_array = np.array([value for value in row[1:]]).reshape(1, -1)

        # Compute the produict
        product = compute_product(x_array,
                                  self._theta_array,
                                  self._biais)

        # Compute the probability
        probability = compute_sigmoid(product)

        # Categorize it
        y_value = 1 if probability >= 0.5 else 0

        # Return the value
        return y_value


    def load_parameters(self, model_version):
        """Load the model's parameters.
        """
        # If a model version is given
        if model_version:

            # Set the folder path
            folder_path = "library/diy/params/" + model_version

            # Load the input dimension
            with open(folder_path + "/input_dim.pkl", "rb") as handle:
                self._input_dim = pickle.load(handle)

            # Load the weights
            with open(folder_path + "/weights.pkl", "rb") as handle:
                self._theta_array = pickle.load(handle)

            # Load the biais
            with open(folder_path + "/biais.pkl", "rb") as handle:
                self._biais = pickle.load(handle)


    def fit(self,
            data_df,
            alpha,
            epochs):
        """Fit the parameters of the model.
        Args:
            data_df (DataFrame): labeled dataset for fitting.
        """
        # Extract input of shape [number_inputs, input_dimension]
        x_array = np.array(data_df.iloc[:, :-1])

        # Extract labeled data from the last column of shape [number_inputs, 1]
        y_array = np.array(data_df.iloc[:, -1:])

        # If starting from nothing
        if self._input_dim is None:

            # Set input dimension
            self._input_dim = x_array.shape[1]

            # Initializa theta
            self._theta_array = np.zeros([1, self._input_dim])

            # Initialize biais
            self._biais = 0

        # Loop through iteration
        for epoch in range(epochs):

            # Compute the cost
            cost = cost_function(x_array,
                                 y_array,
                                 self._theta_array,
                                 self._biais)

            # Display the cost
            print("epoch {} - cost : {}".format(epoch, cost))

            # Compute derivative weights
            derivative_weights, derivative_biais = derivative_cost_function(
                x_array,
                y_array,
                self._theta_array,
                self._biais)

            # Update weights parameters
            self._theta_array += alpha * derivative_weights

            # Display the weights
            # print(self._theta_array)

            # Update biais
            self._biais += alpha * derivative_biais

            # Display the biais
            # print(self._biais)


    # @tools.debug
    def persist_parameters(self, model_version):
        """Persist model's parameters.
        Args:
            model_version (str): version of model to store
        """
        # Set the folder path
        folder_path = "library/diy/params/" + model_version

        # Chack if folder exists
        if not os.path.exists(folder_path):

            # Create the folder
            os.mkdir(folder_path)

        # Persist the input dimension
        with open(folder_path + "/input_dim.pkl", "wb") as handle:
            pickle.dump(self._input_dim, handle)

        # Persist the weights
        with open(folder_path + "/weights.pkl", "wb") as handle:
            pickle.dump(self._theta_array, handle)

        # Persist the biais
        with open(folder_path + "/biais.pkl", "wb") as handle:
            pickle.dump(self._biais, handle)
