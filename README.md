# Reproducible Machine Learning

_A spin off of [cookie-cutter data science](https://github.com/drivendata/cookiecutter-data-science) designed specifically for deep/machine learning project management_

## Overview

Feel like you've been spinning up models endlessly and can't really remember which parameters performed well? Can't find
a clean way to structure a machine learning project when you have multiple different implementations/architectures? 
This repository is aimed at solving your problems. The goal of this initiative is to provide a boilerplate project structure 
when developing machine learning projects. It is designed to be flexible enough to accommodate for the most popular ML
frameworks and problem types, while also providing automated scripts to evaluate, select, and compare models using 
pre-built functions.

## Main Features

1. Clean separation between model architectures in different files
 - These are usually under the `src/models`, where each file contains an architecture
 - In [this](https://github.com/carlomazzaferro/numerai_easy_ml) example, for instance, I stored there a collection
 of Neural Network architectures, such as LSTM, RNN, Fully Connected, etc. Each one lives on an independent file.
2. Boilerplate classes for methods and data structures that are shared between each model
 - Methods such as `fit` and `predict` are common to pretty much any supervised learning task. By providing a set of 
 classes that aim at generalizing these methods for many models, they can be implemented only once, and be shared throguh
 the models using [inheritance](http://www.python-course.eu/python3_inheritance.php)
3. Automatic storage of model parameters in a clean JSON file, predictions, images (ROC curves for instance), 
model dumps 
4. Many other useful scripts to minimize the hassle setting up an environment that enables the user to start working on
what actually matters




Project Organization
--------------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Other data, if needed
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump from the numerai website
    │
    ├── models             <- Trained and serialized models, dumped from TensorFlow, sklearn, etc.
    │
    ├── notebooks          <- Jupyter notebooks for exploration
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── model_ouputs   <- Output of models, which include: ROC curve img, FPR, TPR, paraneters in JSON file,
    │        │                submission file, validation predictions.
    │        ├── MODEL_0
    │        ├── MODEL_1
    │         ...
    │        └── MODEL_N
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling (empty)
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions. Each model spcifies either a Neural Net architecture,
        │   │                 sklearn algorithm, etc.
        │   ├── base.py    <- A general purpose Base and Project class from which the MODEL_N inherits
        │   │                 from, where a variety of useful methods are implemented.
        │   ├── MODEL_1.py
        │   ├── MODEL_2.py
        │   ├── MODEL_3.py
        │   ├── MODEL_4.py
        │   └── __init__.py 
        │     
        └── utils
            ├── __init__.py 
            ├── definitions.py  <- File structure definitions
            └── utilities.py    <- General purpose scripts
    



--------

## Usage

Clone and start implementing your own models. For a starter, you'll want to download the raw data that you'll be using 
to the `/reproducible-ml/data/raw`, and implement your pre-processing steps. These can be implemented in the 
`/reproducible-ml/src/data/make_dataset.py` folder. The template files are pretty much empty, but can be filled with code 
coming for instance from a Jupyter Notebook from where preliminary analyses were being made. `python src/make_dataset.py`
will store your data to `/reproducible-ml/data/processed` so that it can be consumed by your model.

#### Models
Implement your models in the `scr/models` directory. Again, a boilerplate code is provided, and a complete example can be found
 [here](https://github.com/carlomazzaferro/numerai_easy_ml). Ideally, you'd run them simply as `python src/models/MODEL_N.py`, 
 but nothing prevents you from importing them to a Jupyter Notebook and run them there. 
 
#### Stored Data
As mentioned before, model metadata is stored in a JSON file. Using the built-in scripts, as when the method `predict`
is invoked and runs successfully, a new directory will be automatically created in the `src/reports/model_ouptus/` directory.
To get an idea of what the currently implemented scripts do, check out the directory and its contents. 
 
##### TODOs
- Add jupyter notebooks with examples as described above
- Complete visualization code
- Model selection scripts


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
