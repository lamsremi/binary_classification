"""
This script is an implementation of decision tree model.
Decision tree features:
- Attribute selection: Information gain or Gini index?
- Prunning: None, pre or post?
- Categorical or categorical/numerical but always binary
"""

import tools

class DiyDecisionTree():
    """
    Decision tree model
    """

    def __init__(self):
        """
        Inits the decision tree.
        For categorical data
        model = {
            "attribute": "attribute_1",
            "categories": [
                {
                    "value": "value_1",
                    "tree": {
                        "attribute": "attribute_2",
                        "categories": [
                            {
                                "value": "value_1",
                                "tree": {
                                    "label": "label_2"
                                }
                            },
                            {
                                "value": "value_2",
                                "tree": {
                                    "label": "label_1"
                                }
                            }
                        ]
                    }
                },
                {
                    "value": "value_2",
                    "tree": {
                        "label": "label_1"
                    }
                }
            ]
        }
        """
        self.model = self.init_model()


    def init_model(self):
        """Inits the model weights."""
        model = {
            "attribute": "salary",
            "categories": [
                {
                    "value": "1000_5000",
                    "tree": {
                        "attribute": "time",
                        "categories": [
                            {
                                "value": "0_10",
                                "tree": {
                                    "label": "1"
                                }
                            },
                            {
                                "value": "10_50",
                                "tree": {
                                    "attribute": "free",
                                    "categories": [
                                        {
                                            "value": "true",
                                            "tree": {
                                                "label": "0"
                                            }
                                        },
                                        {
                                            "value": "false",
                                            "tree": {
                                                "label": "1"
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                },
                {
                    "value": "0_1000",
                    "tree": {
                        "label": "0"
                    }
                }
            ]
        }

        return model

    # @tools.debug
    def predict(self, input_data=None):
        """
        Predict.
        """
        if input_data is None:
            input_data = {
                "salary": "1000_5000",
                "time": "10_50",
                "free": "false"
            }
        label = None
        model_cursor = self.model

        i_value = 1
        while label is None:
            print("level - {}".format(i_value))
            if "label" in list(model_cursor.keys()):
                label = model_cursor["label"]
                print("  label : {}".format(label))
                break
            else:
                attribute = model_cursor["attribute"]
                value = input_data[attribute]
                model_cursor = next(
                    category for category in model_cursor["categories"] \
                        if category["value"] == value)["tree"]
                print("  {} : {}".format(attribute, value))
            i_value += 1
        return label

    def fit(self, method="gain_information"):
        """
        Fit.
        """
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        stop_condition = False
        while not stop_condition:
            attribute = self.compute_richest_attribute(method)

    def compute_richest_attribute(self, method):
        """
        Compute the richest attribute.
        """
        print("To be coded")

    def compute_information_gain(self):
        """
        Comput the gain of information.
        Args:

        """
        parent_impurity = self.measure_impurity()
        gain =


    def measure_impurity(self, y_array, method="entropy"):
        """
        Compute the information in a given set post spit.
        Args:
            y_array = [0, 1, 1, ...]
        """
        unique_values = list(set(y_array))
        if method == "gini_impurity":
            print("to be coded")
        elif method == "entropy":
            probabilities = []
            for value in unique_values:
                prob = y_array.count(value)
                probabilities.append(-prob*np.log2(prob))
            impurity = np.sum(probabilities)
        return impurity

