"""
Script including the scikit learn model
"""
import pickle

from sklearn.linear_model import LogisticRegression


class ScikitLogisticReg():
    """Class on top of the scikit learn class."""

    def __init__(self):
        """Init class."""
        self.model = LogisticRegression(
            penalty="l2",
            dual=False,
            tol=0.0001,
            C=1.0,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=None,
            solver="liblinear",
            max_iter=100,
            multi_class="ovr",
            verbose=1,
            warm_start=False,
            n_jobs=1)

    def fit(self, x_array, y_array):
        """
        Train the model.
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
        Array standard:
        myArray=[[1,2],[3,4]]
        myArray[0][1] = myArray[1,2][1] = 2
        """
        self.model.fit(x_array, y_array)
        return self.model

    def persist(self, path_sav):
        """Save the model."""
        pickle.dump(self.model, open(path_sav, "wb"))


    def predict(self, x_array):
        """Main functionnality of the model.
        """
        y_array = self.model.predict(x_array)
        return y_array
