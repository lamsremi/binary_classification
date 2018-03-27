"""Control the preprocessing for each of the dataset.
"""
import sys
import importlib


def main(source):
    """Prepare the data to the defined proper format.
    Args:
        source (str): name of the dataset to use.
    """
    # Import the module related to the given data source
    preprocess_module = importlib.import_module("data.{}.preprocess".format(source))
    # Trigger the proprocessing.
    preprocess_module.main()


if __name__ == '__main__':
    # Retrieve the name of the data source from the terminal
    source = sys.argv[1] if len(sys.argv) > 1 else "us_election"
    # Perform the preprocessing
    main(source)
