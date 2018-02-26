# Logistic Regression Implementations


## Introduction

This module presents 2 different implementations of the same logistic regression model :
* The first implementation uses scikit-learn framework.
* The second one is coded from scratch.

The implemented model is a basic logistic regression classifier with the following attributes :
* Stochastic Gradient Descent solver.
* No regularization.
* Cross Entropy as cost function.
* No batch.


## Motivation

The purpose of this exercice is to gain a better understanding on how does logistic regression work.

It is also a good exercice to practice python programming.

## Code structure

The code is structured as follow :

```
pyLogisticRegression
│
├- data/
│   ├- diabetes/
│   ├- kaggle/
│   └- us_election/
│
├- library/
│   ├- doityourself/
│   └- scikit_learn/
│
├- performance/
│   └- num_bench/
│
├- unittest/
│   └- test_core.py
│
├- evaluate.py
├- predict.py
├- prepare.py
├- train.py
│
├- docs/
│   └- Lect09.pdf
│
├- .gitignore
├- README.md
└- requirements.txt
```

## Installation

To use the different implementations, you can directly clone the repository :

```
$ git clone https://github.com/lamsremi/pyLogisticRegression.git
```

### Using a virtual environment

First create the virtual environment :

```
$ python3 -m venv path_to_the_env
```

Activate it :

```
$ source path_to_the_env/bin/activate
```

Then install all the requirements :

```
$ pip install -r requirements.txt
```

## Test

To test if all the functionnalities are working :

```
$ python -m unittest discover -s unittest
```

## Use

For training using a file :

```
>>> form train import main
>>> for source in ["us_election"]:
        for model in ["scikit_learn_sag", "diy"]:
            main(data_df=None,
                 data_source=source,
                 model_type=model,
                 starting_version=None,
                 stored_version="X")
```

Or from the terminal :

```
$ python train.py
```


## Author

Rémi Moise

moise.remi@gmail.com

## License

MIT License

Copyright (c) 2017