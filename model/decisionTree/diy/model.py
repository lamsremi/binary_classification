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
        self.model = self.init_model()


    def init_model(self):
        """Inits the model weights."""
        model = {
            "salary": {
                "type": "numerical",
                "condition_value": 1000,
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
                                "condition_value": True,
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
        return model

    # @tools.debug
    def predict(self, input_data=None):
        """
        Predict.
        """
        if input_data is None:
            input_data = {
                "salary": 500,
                "time": 15,
                "free": False
            }
        label = None
        model_dict = self.model
        i_value = 1
        while label is None:
            print("level - {}".format(i_value))
            if "label" in list(model_dict.keys()):
                label = model_dict["label"]
                print("  label : {}".format(label))
                break
            else:
                attribute = list(model_dict.keys())[0]
                value = input_data[attribute]
                if model_dict[attribute]["type"] == "numerical":
                    condition = value >= model_dict[attribute]["condition_value"]
                    operator = ">=" if condition else "<"
                    print("  {} : {} [{} {}]".format(
                        attribute, value, operator, model_dict[attribute]["condition_value"]))
                if model_dict[attribute]["type"] == "categorical":
                    condition = value == model_dict[attribute]["condition_value"]
                    operator = "==" if condition else "!="
                    print("  {} : {} [{} {}]".format(
                        attribute, value, operator, model_dict[attribute]["condition_value"]))
                model_dict = model_dict[attribute]["class_positive"] \
                    if condition else model_dict[attribute]["class_negative"]
            i_value += 1

        return label

    def fit(self):
        """
        Fit.
        """
        stop_condition = False
        while not stop_condition:
            attribute = compute_richest_attribute()
        print("To be coded.")


