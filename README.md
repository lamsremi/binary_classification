# Logistic Regression Implementations


## Introduction

This module presents 2 different implementations of a unique logistic regression model:
* The first one using the framwork scikit-learn.
* The second one from scratch.

The implemented model is a basic logistic regression model with the following attributes:
* Stochastic Gradient Descent solver
* No regularization
* Cross entropy cost function
* No batch


## Motivation

The purpose of this exercice is to gain a complete understanding of how logistic regression work.

It is also a good exercice to practice and improve python programming skills.

## Code structure

The code is structured as follow:
```
myLogisticRegression
├- docs/
│   └- Lect09.pdf
├- data/
│   ├- diabetes/
│   ├- kaggle/
│   └- us_election/
├- library/
│   ├- doityourself/
│   └- scikit_learn/
├- performance/
│   └- num_bench/
├- unittest/
│   └- test_core.py
├- .gitignore
├- predict.py
├- prepare.py
├- README.md
├- requirements.txt
├- tools.py
└- train.py
```

## Installation

To use the diffrent implementations, you can directly clone the repository :

```
$ git clone https://github.com/lamsremi/pyLogisticRegression.git
```

### Using a virtual environment

Firt create a virtual environment :

```
$ python3 -m venv path_to_the_env
```

The activate it

```
$ source path_to_the_env/bin/activate
```

Then install the requirements :

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

# Or from the terminal :

```
$ python train.py
```


## Author

Rémi Moise

moise.remi@gmail.com

## License

MIT License

Copyright (c) 2017 Rémi Moïse