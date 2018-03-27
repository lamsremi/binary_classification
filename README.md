# Logistic Regression Implementations


## Introduction

This module presents 2 different implementations of the same logistic regression model :
* The first and main implementation is in pure python using only native python 3.6.4 libraries.
* The second one is based on scikit learn framework.

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
│   ├- pure_python/
│   └- scikit_learn/
│
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

## Use

In the terminal

```
$ python train.py pure_python us_election
```
```
$ python train.py <model_type> <data_source>
```


## Author

Rémi Moise

moise.remi@gmail.com

## License

MIT License

Copyright (c) 2017