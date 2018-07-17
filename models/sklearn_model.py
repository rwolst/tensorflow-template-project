"""Defines a class for runnning a softmax regression with Scikit-Learn. This
does not inherit from basic_model.py as it is not a Tensorflow model. Note that
there have been issues in the past with using the LogisticRegression()
multinomial option, when only having 2 output classes.
"""

import os
import re
import copy
import json
import scipy.sparse
import dill as pickle
import numpy as np
import scipy as sp
from pprint import pprint
import matplotlib.pyplot as plt
from contexttimer import Timer

from sklearn.linear_model import LogisticRegression


def get_episode_ids(result_dir):
    """For a given result directory, get all trained episodes inside it."""
    episode_ids = []
    file_names = os.listdir(result_dir)
    for f in file_names:
        temp = re.search('^ep_(\d+).pickle', f)
        if temp:
            episode_ids.append(int(temp.group(1)))

    return episode_ids


class SklearnModel():
    """The class for learning a softmax regression with Scikit-Learn."""

    def __init__(self, config):
        """Only require a configuration to initialise the model.
        Inputs:
            config: A dictionary with model configuration information."""
        # The model with the best hyper-parameters so far, is kept saved.
        # This is a mechanism to load it and override the reset of the
        # configuration.
        if config['best']:
            config.update(self.get_best_config(config['env_name']))

        # Make a `deepcopy` of the configuration before using it to avoid any
        # potential mutation.
        self.config = copy.deepcopy(config)

        if config['debug']:
            print('Config:')
            pprint(self.config)

        # Important to have a random seed for reproducible experimanets.
        # Remember to use it in your TF graph (`tf.set_random_seed()`).
        self.random_seed = self.config['random_seed']

        # In order to reproduce training that is started and stopped, we need
        # to be able to exactly recreate the samples when `method == 'sample'`.
        # Hence we save a sample seed separate to the random seed.
        self.sample_seed = self.config['sample_seed']

        # Copy parameters into model.
        self.result_dir = self.config['result_dir']
        self.data_dir = self.config['data_dir']
        self.max_iter = self.config['max_iter']
        self.tol = self.config['tol']
        ## Penalty weight (may need adjusting to fit with Tensorflow).
        self.C = self.config['C']
        self.R = self.config['R']  # Length of independent variables.
        self.K = self.config['K']  # Total classes.

        if self.K == 2:
            print("Warning: Scikit-Learn has known issues with"
                  " LogisticRegression when using total classes K=2. This is"
                  " certainly the case for warm_start=True and maybe when"
                  " using pred_proba(). It should be fixed in future.")

        self.method = self.config['method']
        self.n_samples = self.config['n_samples']
        if self.method not in ['sample', 'mean', 'true']:
            raise Exception("The method in configuration must be either 'mean'"
                            ", 'sample' or 'true'.")
        elif self.method in ['mean', 'true']:
            if self.n_samples != 1:
                raise Exception("Number of samples can only be 1 when using"
                                " methods 'mean' or 'true'.")
        elif self.method == 'sample':
            if self.n_samples != 1:
                raise Exception("Number of samples can only be 1 when using"
                                " a Scikit-Learn model.")

        # Initialise logistic regression model.
        self.lr = LogisticRegression(penalty='l2',
                                     solver=self.config['solver'],
                                     multi_class='multinomial',
                                     max_iter=self.max_iter,
                                     C=self.C,
                                     fit_intercept=True,
                                     warm_start=True)
        if self.config['debug']:
            print("Using solver: %s." % self.lr.solver)

        self.init()

    def infer(self, X):
        """Predicts probabilities for some input data.
        Inputs:
            X: A (N, R) matrix of independent variables."""
        assert X.shape[1] == self.R

        return self.lr.predict_proba(X)

    def train(self):
        """Train the regression model."""
        # Load data.
        with Timer() as t:
            if self.method == 'true':
                X = np.load(self.data_dir + '/train/X.npy')
            elif self.method == 'mean':
                X = np.load(self.data_dir + '/train/mu.npy')
            elif self.method == 'sample':
                mu = np.load(self.data_dir + '/train/mu.npy')
                L_T = np.load(self.data_dir + '/train/L_T.npy')

                X = np.array(
                    [sp.sparse.linalg.spsolve_triangular(
                        t,
                        np.random.normal(0, 1, self.R).astype(np.float32),
                        lower=False)
                     for t in L_T]).astype(np.float32)

                X = X + mu

        if self.config['debug']:
            print("Time to load/sample data: %s." % t.elapsed)

        Y = np.load(self.data_dir + '/train/Y.npy')

        self.lr.fit(X, Y)
        self.episode_id += 1

        if self.config['debug']:
            # Print the mean log-likelihood on the data.
            pred = self.infer(X)
            LL = np.mean(np.log(pred[np.arange(Y.size), Y]))

            print('Log likelihood on training data: %s' % LL)

    def save(self):
        """Save the model to our results directory by using the episode_id."""
        if self.config['debug']:
            print('Saving to %s with episode_id %s' % (self.result_dir,
                                                       self.episode_id))

        # Build object for saving.
        out = {'random_state': self.random_state,
               'coef_': self.lr.coef_,
               'intercept_': self.lr.intercept_,
               'n_iter_': self.lr.n_iter_}

        # Make the result directory if it doesn't exist.
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        pickle.dump(out, open(self.result_dir +
                              '/ep_%s.pickle' % self.episode_id, 'wb'))

        # Make sure to include the config in the results if it doesn't
        # already exist.
        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(config, f)

    def plot(self):
        """Load the trained episodes and plot their performance on the
        training, validation and test data."""
        episode_ids = np.sort(get_episode_ids(self.result_dir))
        n_episodes = len(episode_ids)

        # Load training, validation and testing data.
        train = {}
        val = {}
        test = {}

        train['X'] = np.load(self.data_dir + '/train/X.npy')
        train['Y'] = np.load(self.data_dir + '/train/Y.npy')
        n_train = train['X'].shape[0]

        val['X']   = np.load(self.data_dir + '/val/X.npy')
        val['Y']   = np.load(self.data_dir + '/val/Y.npy')
        n_val = val['X'].shape[0]

        test['X']  = np.load(self.data_dir + '/test/X.npy')
        test['Y']  = np.load(self.data_dir + '/test/Y.npy')
        n_test = test['X'].shape[0]

        # Get info for plotting.
        train['cross_entropy'] = np.empty(n_episodes)
        val['cross_entropy']   = np.empty(n_episodes)
        test['cross_entropy']  = np.empty(n_episodes)
        for i, ep in enumerate(episode_ids):
            ## Load the Logisitic Regression internal state.
            out = pickle.load(open(self.result_dir +
                                   '/ep_%s.pickle' % ep, 'rb'))

            self.lr.intercept_ = out['intercept_']
            self.lr.coef_ = out['coef_']
            self.lr.n_iter_ = out['n_iter_']

            ## Get predictions.
            train['pred'] = self.infer(train['X'])
            val['pred']   = self.infer(val['X'])
            test['pred']  = self.infer(test['X'])

            ## Calculate mean cross entropy.
            train['cross_entropy'][i] = np.mean(
                np.log(train['pred'][np.arange(n_train), train['Y']]))
            val['cross_entropy'][i]   = np.mean(
                np.log(val['pred'][np.arange(n_val), val['Y']]))
            test['cross_entropy'][i]  = np.mean(
                np.log(test['pred'][np.arange(n_test), test['Y']]))

        # Make the plots.
        plt.plot(episode_ids, train['cross_entropy'])
        plt.plot(episode_ids, val['cross_entropy'])
        plt.plot(episode_ids, test['cross_entropy'])
        plt.legend(['Train', 'Validation', 'Test'])

        plt.show()

    def init(self):
        """Reload a model if it already exists otherwise initialise with
        random values."""
        if not os.path.exists(self.result_dir):
            # Set random states.
            self.random_state = np.random.RandomState(self.random_seed)

            ## Set the initial parameters to random values so that we can call
            ## `infer()` even without `train`.
            self.lr.coef_ = self.random_state.randn(self.K, self.R)
            self.lr.intercept_ = self.random_state.randn(self.K)

            # Set the episode_id measuring how many times we have trained the
            # model. As we use `warm_start = True` this is analagous to
            # re-running training in Tensorflow.
            self.episode_id = 0
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)

            # Load latest episode!
            ## Begin by getting all possible episodes.
            episode_ids = get_episode_ids(self.result_dir)

            ## Now load the maximum.
            out = pickle.load(open(self.result_dir +
                                   '/ep_%s.pickle' % max(episode_ids), 'rb'))

            # Set the model with the loaded state.
            self.random_state = out['random_state']
            self.lr.intercept_ = out['intercept_']
            self.lr.coef_ = out['coef_']
            self.lr.n_iter_ = out['n_iter_']

            # Set the episode id.
            self.episode_id = max(episode_ids)

        # Sample state always initialised to the same according to seed.
        self.sample_state = np.random.RandomState(self.sample_seed)

        # Set random state in LogisticRegression model (I believe its ok to
        # set it as no processing is done in LogisticRegression.__init__).
        self.lr.random_state = self.random_state
