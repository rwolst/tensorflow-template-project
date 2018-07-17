"""An endpoint to run the machine learning stuff from."""

import os
import argparse
import numpy as np
import time

from models import (make_model)

# Use absolute paths to avoid any issues.
project_dir = os.path.dirname(os.path.realpath(__file__))

# Create argument parser.
parser = argparse.ArgumentParser(description="Endpoint for running tests on"
                                             " the softamx regression with"
                                             " random inputs.")
parser.add_argument('function',
                    type=str,
                    default="Train",
                    help="What you would like to do: Train, Infer, Plot.")
parser.add_argument('--model_name',
                    type=str,
                    default="TF_Random",
                    help="Name of the model to use.")
parser.add_argument('--best',
                    action='store_true',
                    help="Use best known configuration.")
parser.add_argument('--debug',
                    action='store_true',
                    help="Print the configuration when creating model.")
parser.add_argument('--sample_seed',
                    type=int,
                    default=np.random.randint(0, 2**32 - 1),
                    help="Value of sample seed (used for sampling).")
parser.add_argument('--random_seed',
                    type=int,
                    default=np.random.randint(0, 2**32 - 1),
                    help="Value of random seed (used for everything other than"
                         " sampling).")
parser.add_argument('--result_dir',
                    type=str,
                    default=None,
                    help="Name of the directory to store/log the model (if it"
                         " exists, the model will be loaded from it).")
parser.add_argument('--data_dir',
                    type=str,
                    default=project_dir + '/data/',
                    help="Name of the directory data is stored in.")
parser.add_argument('--solver',
                    type=str,
                    default='saga',
                    help="The LogisticRegression solver to use.")
parser.add_argument('--max_iter',
                    type=int,
                    default=2000,
                    help="Number of training steps.")
parser.add_argument('--tol',
                    type=float,
                    default=0,
                    help="Tolerance for convergence. Defaults to 0 so we do"
                         " maximum iterations.")
parser.add_argument('--C',
                    type=float,
                    default=1,
                    help="L2 penalty size.")
parser.add_argument('--R',
                    type=int,
                    default=50,
                    help="Size of independent variables (default should be"
                         " correct).")
parser.add_argument('--K',
                    type=int,
                    default=20,
                    help="Total classes for dependent variables (default"
                         " should be correct).")
parser.add_argument('--method',
                    type=str,
                    default='mean',
                    help="Either 'mean', 'sample' or 'true'. Chooses whether"
                         " to learn by taking mean values from distribution,"
                         " by sampling them, or by using the actual true"
                         " values.")
parser.add_argument('--n_samples',
                    type=int,
                    default=1,
                    help="Total samples to take from distribution. It must be"
                         " 1 when using sklearn model.")

FLAGS, unparsed = parser.parse_known_args()

# Need to set deault results directory after flags are already parsed.
if FLAGS.result_dir is None:
    FLAGS.result_dir = project_dir + '/results/' + \
        FLAGS.model_name + '/' + str(int(time.time()))


if __name__ == "__main__":
    config = FLAGS.__dict__.copy()
    model = make_model(config)

    if config['function'] == 'Train':
        print("Starting %s training." % config['model_name'])

        model.train()
        model.save()

    elif config['function'] == 'Plot':
        # Load all episodes and plot the training, validation and test results.
        print("Starting %s plotting." % config['model_name'])

        model.plot()
