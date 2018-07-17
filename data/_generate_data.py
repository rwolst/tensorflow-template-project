"""A file for generating data for the softmax regression with random
independent varaibles. The constants refer to:
    1) N_TRAIN, N_VAL, N_TEST: Total number of training, validation and test
                               samples.
    2) R: The dimension of the independent variables (X).
    3) K: The total classes that the dependent variables (Y) can be e.g. for
          logisitic regression this is 2. For softmax this can be any K >= 2.

The actual model is defined by firstly the independent variables distribution
    X ~ MVN(mu, sigma)
then the multinomial probabilities
    P = b + X @ W
and finally the dependent variable distribution
    Y = Multinomial(P).

The genereted files are:
    1) train/mu.csv: A (N_TRAIN x R) float32 matrix containing the mean of the
                     independent variables for the training set.
    2) val/mu.csv:   A (N_VAL x R) float32 matrix containing the mean of the
                     independent variables for the validation set.
    3) test/mu.csv:  A (N_TEST x R) float32 matrix containing the mean of the
                     independent variables for the test set.
    4) train/sigma.csv: A (N_TRAIN x R x R) float32 tensor such that
                            sigma[i,:,:]
                        is the covariance of the independent variables at
                        observation i of the training set.
    5) val/sigma.csv:   A (N_VAL x R x R) float32 tensor such that
                            sigma[i,:,:]
                        is the covariance of the independent variables at
                        observation i of the validation set.
    6) test/sigma.csv:  A (N_TEST x R x R) float32 tensor such that
                            sigma[i,:,:]
                        is the covariance of the independent variables at
                        observation i of the test set.
    7) train/X.csv: A (N_TRAIN x R) float32 matrix containing the true (i.e.
                    sampled from MVN(mu, sigma)) independent variables for the
                    training set.
    8) val/X.csv:   A (N_VAL x R) float32 matrix containing the true (i.e.
                    sampled from MVN(mu, sigma)) independent variables for the
                    validation set.
    9) test/X.csv:  A (N_TEST x R) float32 matrix containing the true (i.e.
                    sampled from MVN(mu, sigma)) independent variables for the
                    test set.
    10) train/Y.csv: A (N_TRAIN,) int32 vector containing the values of the
                     dependent varaible labels for the training set. It can
                     take values on the set [0, ..., K-1].
    11) val/Y.csv:   A (N_VAL,) int32 vector containing the values of the
                     dependent varaible labels for the validation set. It can
                     take values on the set [0, ..., K-1].
    12) test/Y.csv:  A (N_TEST,) int32 vector containing the values of the
                     dependent varaible labels for the test set. It can
                     take values on the set [0, ..., K-1].
    13) W.csv: A (R, K) float32 matrix. See above for more details.
    14) b.csv: A (K,) float32 vector. See above for more details.
"""

import numpy as np
from contexttimer import Timer
from scipy.stats import ortho_group
import scipy.sparse
import scipy as sp

# We use the following constants for generating the data.
N_TRAIN = 650000
N_VAL   = 150000
N_TEST  = 200000
N = {'train': N_TRAIN, 'val': N_VAL, 'test': N_TEST}
R = 50
K = 20
RANDOM_SEED = 42

# A parameter for controlling how many covariance matrices we sample. As it
# takes a long time, we don't want to have to sample one for every observation.
ORTHO_SAMPLES = 200

# A parameter for controlling how large the values of the covariance matrix are
# and hence how random the independent variables are.
VAR_SIZE = 0.1

# A parameter for controlling density of inverse covariance cholesky factor.
DENSITY = 0.1  # Percentage of non-zero values.


def softmax(Z):
    """A useful function for performing numerically stable vectorised softmax.
    Inputs:
        Z: A (N, K) matrix such that N is the number of observations and
           K is the number of classes. Hence we are performing softmax on
           each Z[i,:].
    """
    # As we are free to add and subtract constants to each row without
    # affecting final result, we transform Z so that it has maximum value of 0.
    Z_max = Z.max(1)
    Z_trans = Z - Z_max[:, None]

    # Now perform softmax.
    Z_trans = np.exp(Z_trans)
    out = Z_trans/Z_trans.sum(1)[:, None]

    return out


def sparse_random(n_rows, n_cols, density, dtype):
    """A faster implementation of scipy.sparse.rand?"""
    if density == 0:
        return sp.sparse.coo_matrix((n_rows, n_cols), dtype=dtype)
    N = n_rows * n_cols
    nnz = int(N * density)
    idx = np.random.choice(N, nnz, replace=False)
    rows, cols = np.divmod(idx, n_cols)
    data = np.random.rand(nnz).astype(dtype)
    return sp.sparse.coo_matrix((data, (rows, cols)), shape=[n_rows, n_cols])


# Set the seed.
np.random.seed(RANDOM_SEED)

# Parameter matrices.
W = np.random.randn(R, K).astype(np.float32)
b = np.random.randn(K).astype(np.float32)

# Multivariate normal distribution i.e. mu and sigma (represented by Cholesky
# factor of its inverse).
mu = {}
L_T = {}

for set_name in N:
    ## Mu is easy.
    mu[set_name] = np.random.randn(N[set_name], R).astype(np.float32)

    ## We don't save the covariance sigma itself, but the lower Cholesky
    ## factor of its inverse. The reason we do this is so that we can assume
    ## the inverse is sparse (which means non-inverse can be dense) and
    ## therefore has sparse Cholesky factor. Denote this factor L s.t.
    ##     L @ L.T = inv(sigma)
    ## and hence
    ##     inv(L.T) @ inv(L) = sigma.
    ## Then we can sample MVN by finding
    ##    mu + inv(L.T) @ z
    ## where z is a vector of independent standard normal samples.
    ## Hence we can simply store the sparse L.T for each observation.
    ## Note also that to calculate
    ##     inv(L.T) @ z,
    ## we do not have to find exact inverse, just solve system of linear
    ## equations
    ##     L.T @ x = z.

    ## We add identity to avoid dealing with singular covariance. Note it is
    ## actually fine for this to be the case (i.e. we have a non-random
    ## feature) but requires in the traingular solve, substituting a 0
    ## value for any of these features, which is not done automatically
    ## (instead an exception is thrown).
    with Timer() as t:
        L_T[set_name] = [sp.sparse.identity(R, dtype=np.float32) +
                         sp.sparse.triu(
                             sparse_random(R, R, DENSITY, np.float32))
                         for i in range(N[set_name])]
    print('Time to generate %s L_T: %s' % (set_name, t.elapsed))

    assert len(L_T[set_name]) == N[set_name]
    assert L_T[set_name][0].shape == (R, R)
    assert L_T[set_name][0].dtype == np.float32

    assert mu[set_name].dtype == np.float32

# Sample the true independent variables X (not sure how to do without a for
# loop but should be fast compared to the sigma generation).
X = {}
for set_name in N:
    X[set_name] = np.empty([N[set_name], R]).astype(np.float32)
    with Timer() as t:
        # As we are using float32 we reduce the tolerance of the positive
        # (semi-)definite check.
        X[set_name] = np.array(
            [sp.sparse.linalg.spsolve_triangular(
                t, np.random.normal(0, 1, R).astype(np.float32), lower=False)
             for t in L_T[set_name]]).astype(np.float32)

        X[set_name] = X[set_name] + mu[set_name]
    print('Time to sample %s X: %s' % (set_name, t.elapsed))

    assert X[set_name].shape == (N[set_name], R)
    assert X[set_name].dtype == np.float32

# Sample the dependent variables Y.
Y = {}
for set_name in N:
    with Timer() as t:
        activations = b + (X[set_name] @ W)
        P = softmax(activations)

        ## Cumulative sum and correct for numerical issues not causing them to
        ## sum to 1.
        P_cum = np.cumsum(P, axis=1).reshape([N[set_name], K])
        P_cum = P_cum / (P_cum[:, -1][:, None])

        U = np.random.rand(N[set_name], 1)

        Y[set_name] = np.sum(P_cum < U, axis=1).astype(np.int32)
    print('Time to sample %s Y: %s' % (set_name, t.elapsed))

    assert np.unique(Y[set_name]).size == K
    assert (np.unique(Y[set_name]) == np.arange(K)).all()
    assert Y[set_name].dtype == np.int32

# Finally we can save all the values. We don't use csv as it is slightly
# awkward for 3D arrays.
np.save("train/mu", mu['train'])
np.save("val/mu", mu['val'])
np.save("test/mu", mu['test'])
np.save("train/L_T", L_T['train'])
np.save("val/L_T", L_T['val'])
np.save("test/L_T", L_T['test'])
np.save("train/X", X['train'])
np.save("val/X", X['val'])
np.save("test/X", X['test'])
np.save("train/Y", Y['train'])
np.save("val/Y", Y['val'])
np.save("test/Y", Y['test'])
np.save("W", W)
np.save("b", b)
