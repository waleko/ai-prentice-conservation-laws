# Beyond Dynamics: Learning to Discover Conservation Principles

This repository is the official implementation of [Beyond Dynamics: Learning to Discover Conservation Principles]().

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Generating Data

To generate data for the experiments in this paper, refer to [peterparity/conservation-laws-manifold-learning](https://github.com/peterparity/conservation-laws-manifold-learning) and `utils/trajectory_generation.py`.

## Training and Evaluation

To train and evaluate the model(s) in the paper, take a look at the jupyter notebooks in the repository.

- `find_early_stopping_parameters.ipynb` contains the code to find the threshold
- `noise.ipynb` contains the code to train and evaluate the model with noisy data and determine the maximum noise level
- `prentice_*` notebooks contain the code to train and evaluate the models using experiment data referenced in the paper.
