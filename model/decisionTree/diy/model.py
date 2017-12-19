"""
This script is an implementation of decision tree model.
Decision tree features:
- Attribute selection: Information gain or Gini index?
- Prunning: None, pre or post?
- Categorical or categorical/numerical but always binary
"""


class DiyDecisionTree():
    """
    Decision tree model
    """

    def __init__(self):
        """
        Inits the decision tree.
        model = {
            "attribute_1[salary]": {
                "type": "numerical",
                "condition_value": 20000,
                "class_positive":{
                    "label": 1
                },
                "class_negative":{
                    "attribute_2[time]": {
                        "type": "numerical",
                        "condition_value": 1,
                        "class_positive": {
                            "attribute_3[yes/no]": {
                                "type": "categorical",
                                "class_positive": {
                                    "label": 0
                                },
                                "class_negative": {
                                    "label": 1
                                }
                            }
                        },
                        "class_negative": {
                            "label": 0
                        }
                    }
                }
            }
        }
        """


    def init_model():
        """Inits the model weights."""
        self.model = {
            "salary": {
                "type": "numerical",
                "condition_value": "1000",
                "class_positive":{
                    "label": 1
                },
                "class_negative":{
                    "time": {
                        "type": "numerical",
                        "condition_value": 10,
                        "class_positive": {
                            "free": {
                                "type": "categorical",
                                "condition_value": true
                                "class_positive": {
                                    "label": 0
                                },
                                "class_negative": {
                                    "label": 1
                                }
                            }
                        },
                        "class_negative": {
                            "label": 0
                        }
                    }
                }
            }
        }

    def predict(self):
        """
        Predict.
        """
        input_data = {
            "salary": 500,
            "time": 15,
            "free": False
        }
        label = None
        model_dict = self.model
        wihle label is None:
            if "label" in list(model_dict.keys()):
                label = model_dict["label"]
                break
            else:
                attribute = model_dict.keys()[O]
                value = input_data[attribute]
                if self.model[attribute]["type"] == "numerical":
                    condition = value >= self.model[attribute]["condition_value"]
                if self.model[attribute]["type"] == "categorical":
                    condition = value == self.model[attribute]["condition_value"]
                model_dict = model_dict[attribute]["class_positive"] if condition: else model_dict[attribute]["class_negative"]
        return label

    def fit(self):
        """
        Fit.
        """
        print("To be coded.")

