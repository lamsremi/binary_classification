"""Module for testing the core functionnalities.
"""
import unittest

import pandas as pd

import train
import predict


class TestCore(unittest.TestCase):
    """Core test class.
    """
    def test_scikitlearn_sag(self):
        """Perform the test of the scikit-learn implementation.
        """
        # Load a sample of data
        data_df = pd.read_csv("data/us_election/data.csv",
                              nrows=20)
        # Train a model with this dataset
        self.assertTrue(train.main(train_data=data_df,
                                   dataset=None,
                                   model_type="scikit_learn_sag",
                                   start_version=None,
                                   end_version="unittest"))

        # Predict
        prediction_df = predict.main(inputs_data=data_df.iloc[0:10, :-1],
                                     model_type="scikit_learn_sag",
                                     model_version="unittest")
        # CHeck if not None
        for row in prediction_df.itertuples():
            self.assertIsNotNone(row.prediction)

    def test_diy(self):
        """Perform the test of the scikit-learn implementation.
        """
        # Load a sample of data
        data_df = pd.read_csv("data/us_election/data.csv",
                              nrows=20)
        # Train a model with this dataset
        self.assertTrue(train.main(train_data=data_df,
                                   dataset=None,
                                   model_type="diy",
                                   start_version=None,
                                   end_version="unittest"))

        # Predict
        prediction_df = predict.main(inputs_data=data_df.iloc[0:10, :-1],
                                     model_type="diy",
                                     model_version="unittest")
        # CHeck if not None
        for row in prediction_df.itertuples():
            self.assertIsNotNone(row.prediction)
