"""Contains a Tensorflow model for solving the optimisation problem."""

import scipy.sparse
import numpy as np
import scipy as sp
import tensorflow as tf
from contexttimer import Timer

from ._tf_basic_model import TFBasicModel


class TFModel(TFBasicModel):
    """A Tensorflow model for optimising the softmax regression with known
    independent variable distributions."""

    def set_model_props(self):
        """Set specific properties for the optimisation problem."""
        self.R = self.config['R']  # Length of independent variables.
        self.K = self.config['K']  # Total classes.
        self.C = self.config['C']  # Regularisation weight.

        self.method = self.config['method']
        self.n_samples = self.config['n_samples']
        if self.method not in ['sample', 'mean', 'true']:
            raise Exception("The method in configuration must be either 'mean'"
                            ", 'sample' or 'true'.")
        elif self.method in ['mean', 'true']:
            if self.n_samples != 1:
                raise Exception("Number of samples can only be 1 when using"
                                " methods 'mean' or 'true'.")

    def build_graph(self):
        graph = tf.Graph()
        with graph.as_default():
            # We treat the independent varaibles as X (even though it can come
            # from the means, the true values, or sampled values).
            X = tf.placeholder(tf.float32,
                               shape=[self.n_samples, None, self.R],
                               name='X')

            # The depedent variables are the same no matter what and treated as
            # Y.
            Y = tf.placeholder(tf.float32,
                               shape=[None, self.K],
                               name='Y')

            # The regularisation weight should be the same as in scikit-learn
            # LogisticRegression(), if it is set to (1/C)*(1/N), where
            #     C: The value passed to LogisticRegression().
            #     N: The total size of the whole training data (not just this
            #        batch).
            # Note this needs double checking.
            alpha = tf.placeholder(tf.float32, shape=(), name='alpha')

            # The parameters are biases (b) and weight matrix (W).
            W = tf.get_variable('W', shape=[self.R, self.K], dtype=tf.float32,
                                initializer=tf.random_normal_initializer)
            b = tf.get_variable('b', shape=[1, self.K], dtype=tf.float32,
                                initializer=tf.random_normal_initializer)

            # Calculate predictions (note this requires repeating W along
            # a new 0th axis, so that it matches with X).
            pred = tf.nn.softmax(b + tf.matmul(X, tf.tile(tf.expand_dims(W, 0),
                                               [self.n_samples, 1, 1])))

            # As per the latex document, get the mean of the predictions over
            # all samples, for each observation. This has shape [None, K],
            # where None represents the batch size.
            pred_mean = tf.reduce_mean(pred, axis=0)

            # Multiply this with true values and reduce sum to get the pre-log
            # values for the loss (should be shape [None]).
            pre_log_likelihood = tf.reduce_sum(tf.multiply(pred_mean, Y),
                                               axis=1)

            # Finally get true log likelihood by logging and taking the
            # mean.
            log_likelihood = tf.reduce_mean(tf.log(pre_log_likelihood),
                                            name='log-likelihood')

            # The loss function is now the -log_likelihood plus the regulariser
            # term.
            regulariser = 0.5 * alpha * tf.reduce_sum(tf.square(W))
            loss = tf.add(-log_likelihood, regulariser, name='loss')

            # Given the loss function, we can optimise using a Tensorflow
            # optimiser.
            opt = tf.train.AdamOptimizer()
            gradients = opt.compute_gradients(loss)
            opt.apply_gradients(gradients, name='train-step')

            # Initialise operation should be stored with name 'init'.
            tf.global_variables_initializer()

        return graph

    def load_data(self, data_set):
        """Load data depending on the model configuration.
        Inputs:
            data_set: One of 'train', 'val' or 'test'.
            """
        with Timer() as t:
            if self.method == 'true':
                X = np.load(self.data_dir + '/%s/X.npy' % data_set)

                # Add a preceeding dimension (to represent sample of 1).
                X = X[None, ...]
            elif self.method == 'mean':
                X = np.load(self.data_dir + '/%s/mu.npy' % data_set)

                # Add a preceeding dimension (to represent sample of 1).
                X = X[None, ...]
            elif self.method == 'sample':
                mu = np.load(self.data_dir + '/%s/mu.npy' % data_set)
                L_T = np.load(self.data_dir + '/%s/L_T.npy' % data_set)

                X = np.array(
                    [sp.sparse.linalg.spsolve_triangular(
                        t,
                        np.random.randn(self.R,
                                        self.n_samples).astype(np.float32),
                        lower=False)
                     for t in L_T]).astype(np.float32)

                if self.n_samples == 1:
                    X = X[None, ...]
                else:
                    X = np.transpose(X, [2, 0, 1])

                X = X + mu

        if self.config['debug']:
            print("Time to load/sample %s data: %s." % (data_set, t.elapsed))

        # Tensorflow expects labels as (N, K) matrix of 1s and 0s but our
        # data is a (N,) vector of class label integers.
        Y = np.load(self.data_dir + '/%s/Y.npy' % data_set)
        N = Y.shape[0]
        Y_tf = np.zeros([N, self.K]).astype(np.float32)
        Y_tf[np.arange(N), Y] = 1

        # Make sure X is (n_samples, N, R)
        assert X.shape == (self.n_samples, N, self.R)

        return X, Y_tf

    def learn_from_epoch(self, data):
        """Train the model of 1 epoch (pass through training data)."""
        # Separate data.
        X, Y = data
        n_samples, n_obs, _ = X.shape

        # Set the regularisation strength (I believe this is correct).
        alpha = (1 / n_obs) * (1 / self.C)

        # Reset the batch index.
        batch_index = 0

        # Find graph operations and tensors we will need to call.
        train_op = self.graph.get_operation_by_name('train-step')
        log_likelihood_tensor = self.graph.get_tensor_by_name(
            'log-likelihood:0')
        loss_tensor = self.graph.get_tensor_by_name('loss:0')

        with Timer() as t:
            # Shuffle data.
            perm = np.arange(n_obs)
            np.random.shuffle(perm)

            if self.config['debug']:
                print('Time to shuffle data: %s' % t.elapsed)

        with Timer() as t:
            # Loop over the batches.
            for i in range(int(n_obs / self.batch_size)):
                ## Get the batches for training
                start = batch_index
                end = batch_index + self.batch_size
                idx = perm[start:end]

                # Slice the batches.
                X_batch = X[:, idx, :]
                Y_batch = Y[idx, :]

                # Run throught the graph.
                self.sess.run(train_op, feed_dict={'X:0': X_batch,
                                                   'Y:0': Y_batch,
                                                   'alpha:0': alpha})

                batch_index += self.batch_size

            if self.config['debug']:
                print('Time for epoch training: %s' % t.elapsed)

        # Get training and validation error after epoch.
        if self.config['debug']:
            with Timer() as t:
                log_likelihood, loss = self.sess.run(
                    [log_likelihood_tensor, loss_tensor],
                    feed_dict={'X:0': X, 'Y:0': Y, 'alpha:0': alpha})

                print("(Log likelihood, loss, inference time):"
                      " ({}, {}, {})".format(log_likelihood,
                                             loss,
                                             t.elapsed))
