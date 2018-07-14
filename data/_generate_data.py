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

# We use the following constants for generating the data.
N_TRAIN = 650000
N_VAL   = 150000
N_TEST  = 200000
N = {'train': N_TRAIN, 'val': N_VAL, 'test': N_TEST}
R = 20
K = 20
RANDOM_SEED = 42

# A parameter for controlling how many covariance matrices we sample. As it
# takes a long time, we don't want to have to sample one for every observation.
ORTHO_SAMPLES = 200

# A parameter for controlling how large the values of the covariance matrix are
# and hence how random the independent variables are.
VAR_SIZE = 0.1


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


# Set the seed.
np.random.seed(RANDOM_SEED)

# Parameter matrices.
W = np.random.randn(R, K).astype(np.float32)
b = np.random.randn(K).astype(np.float32)

# Multivariate normal distribution i.e. mu and sigma.
mu = {}
sigma = {}

for set_name in N:
    ## Mu is easy.
    mu[set_name] = np.random.randn(N[set_name], R).astype(np.float32)

    ## Sigma we have to ensure is symmetric positive definite.
    ## Note however, we can just create a lower traingular matrix as this is
    ## all thats needed for sampling.
    ## Note also that we cannot store a huge covariance matrix, so to get more
    ## samples we can either make it sparse, or only generate a small amount
    ## and index them for each observation.
    ##     X = (np.tril(np.random.rand(5, N, N)) @ np.random.multivariate_normal(np.zeros(N), np.eye(N), size = 5)[:,:,None])

    ### Sample orthogonal matrices.
    with Timer() as t:
        Q = ortho_group.rvs(dim=R, size=ORTHO_SAMPLES).astype(np.float32)
        idx = np.random.choice(ORTHO_SAMPLES, N[set_name], replace=True)
        Q = Q[idx,:,:]
    print('Time to generate %s Q: %s' % (set_name, t.elapsed))

    assert Q.shape == (N[set_name], R, R)

    ### Build diagonal matrix for controlling covariance entry size.
    ### Note that we add a small value, as even though the eignevalues are
    ### positive then numerical issues when they are very small can cause them
    ### to become negative when.
    with Timer() as t:
        D = VAR_SIZE * np.apply_along_axis(
                np.diag, 1, 0.001 + np.random.rand(N[set_name], R)
            ).astype(np.float32)
    print('Time to generate %s D: %s' % (set_name, t.elapsed))

    ### Combine to create our symmetric positive definite sigma matrices i.e.
    ###     sigma[i,:,:] = Q[i,:,:] @ D[i,:,:] @ Q[i,:,:].T
    ### as symmetric positive definite matrices have orthogonal eigenvectors
    ### and strictly positive eignevalues.
    with Timer() as t:
        sigma[set_name] = np.matmul(np.matmul(Q, D),
                                    np.transpose(Q, [0, 2, 1]))
    print('Time to generate %s sigma: %s' % (set_name, t.elapsed))

    # Make sure the first eigenvalues are correct.
    assert ((np.sort(np.linalg.eig(sigma[set_name][0,:,:])[0]) -
             np.sort(np.diag(D[0,:,:])))**2 < 1e-5).all()

    assert mu[set_name].dtype == np.float32
    assert sigma[set_name].dtype == np.float32

# Sample the true independent variables X (not sure how to do without a for
# loop but should be fast compared to the sigma generation).
X = {}
for set_name in N:
    X[set_name] = np.empty([N[set_name], R]).astype(np.float32)
    with Timer() as t:
        for i in range(N[set_name]):
            # As we are using float32 we reduce the tolerance of the positive
            # (semi-)definite check.
            X[set_name][i,:] = np.random.multivariate_normal(
                                   mu[set_name][i,:],
                                   sigma[set_name][i,:,:],
                                   size=1,
                                   tol=1e-6)
    print('Time to sample %s X: %s' % (set_name, t.elapsed))

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
np.save("train/sigma", sigma['train'])
np.save("val/sigma", sigma['val'])
np.save("test/sigma", sigma['test'])
np.save("train/X", X['train'])
np.save("val/X", X['val'])
np.save("test/X", X['test'])
np.save("train/Y", Y['train'])
np.save("val/Y", Y['val'])
np.save("test/Y", Y['test'])
np.save("W", W)
np.save("b", b)
