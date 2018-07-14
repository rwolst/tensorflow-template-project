import os
import copy
from models import (make_model)

# Use absolute paths to avoid any issues.
project_dir = os.getcwd()


def test_sklearn_max_iter():
    """Test that a SklearnModel run with 2 episodes of max_iter, gives the
    same results as when run with one episode of 2*max_iter, with the same
    starting seed.

    Note this does not work for sag and saga algorithms due to the internal
    state of the optimisation not being saved between runs."""
    result_dir = project_dir + '/results/tests/'

    config1 = {'model_name': 'SklearnModel',
               'best': False,
               'debug': False,
               'random_seed': 42,
               'result_dir': result_dir + '1',
               'data_dir': project_dir + '/data',
               'max_iter': 10,
               'tol': 0,
               'C': 1,
               'R': 5,
               'K': 3,
               'method': 'mean',
               'n_samples': 1,
               'solver': 'newton-cg'}

    # Train model1 twice.
    model1 = make_model(config1)
    model1.train()
    model1.train()

    # Train model2 once (but with twice the iterations).
    config2 = copy.deepcopy(config1)
    config2['max_iter'] = 20
    model2 = make_model(config2)
    model2.train()

    # Make sure they give same results.
    assert (model1.lr.intercept_ == model2.lr.intercept_).all()
    assert (model1.lr.coef_ == model2.lr.coef_).all()


def test_sklearn_repeated():
    """Test two runs with the same random seed give the same results."""
    result_dir = project_dir + '/results/tests/'

    config1 = {'model_name': 'SklearnModel',
               'best': False,
               'debug': False,
               'random_seed': 42,
               'result_dir': result_dir + '1',
               'data_dir': project_dir + '/data',
               'max_iter': 10,
               'tol': 0,
               'C': 1,
               'R': 5,
               'K': 3,
               'method': 'mean',
               'n_samples': 1,
               'solver': 'saga'}

    # Train model1.
    model1 = make_model(config1)
    model1.train()

    # Train model2.
    config2 = copy.deepcopy(config1)
    model2 = make_model(config2)
    model2.train()

    # Make sure they give same results.
    assert (model1.lr.intercept_ == model2.lr.intercept_).all()
    assert (model1.lr.coef_ == model2.lr.coef_).all()
    assert (model1.lr.n_iter_ == model2.lr.n_iter_).all()
