"""Prepare data script.
"""
import importlib


def main(data_source):
    """Prepare the data from the data folder.
    Args:
        data_source (str): source of the data to use.
    """
    # Import
    preprocess_module = importlib.import_module("data.{}.preprocess".format(data_source))

    # Preprocess the data
    preprocess_module.main()


if __name__ == '__main__':
    for source in ["diabete", "kaggle", "us_election"]:
        main(data_source=source)
