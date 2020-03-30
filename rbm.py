"""Restricted Boltzmann Machine
"""

# Authors: Yann N. Dauphin <dauphiya@iro.umontreal.ca>
#          Vlad Niculae
#          Gabriel Synnaeve
#          Lars Buitinck
# License: BSD 3 clause

import time

import numpy as np
import scipy.sparse as sp
from scipy.special import expit  # logistic function
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import gen_even_slices
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import log_logistic
from sklearn.utils.validation import check_is_fitted
from d_wave_client import *


class BernoulliRBM(TransformerMixin, BaseEstimator):
    """Bernoulli Restricted Boltzmann Machine (RBM).

    A Restricted Boltzmann Machine with binary visible units and
    binary hidden units. Parameters are estimated using Stochastic Maximum
    Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
    [2].

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Read more in the :ref:`User Guide <rbm>`.

    Parameters
    ----------
    n_components : int, default=256
        Number of binary hidden units.

    learning_rate : float, default=0.1
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, default=10
        Number of examples per minibatch.

    n_iter : int, default=10
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : int, default=0
        The verbosity level. The default, zero, means silent mode.

    random_state : integer or RandomState, default=None
        Determines random number generation for:

        - Gibbs sampling from visible and hidden layers.

        - Initializing components, sampling from layers during fit.

        - Corrupting the data when scoring samples.

        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    intercept_hidden_ : array-like, shape (n_components,)
        Biases of the hidden units.

    intercept_visible_ : array-like, shape (n_features,)
        Biases of the visible units.

    components_ : array-like, shape (n_components, n_features)
        Weight matrix, where n_features in the number of
        visible units and n_components is the number of hidden units.

    h_samples_ : array-like, shape (batch_size, n_components)
        Hidden Activation sampled from the model distribution,
        where batch_size in the number of examples per minibatch and
        n_components is the number of hidden units.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import BernoulliRBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = BernoulliRBM(n_components=2)
    >>> model.fit(X)
    BernoulliRBM(n_components=2)

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008
    """
    op_mode_classic = 0
    op_mode_quantum = 1
    op_mode_simulate_quantum = 2

    def __init__(self, tester, n_components=256, learning_rate=0.1, batch_size=10,
                 n_iter=10, verbose=0, random_state=None, op_mode=0, ):
        self.tester = tester
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        self.op_mode = op_mode

    def set_op_mode(self, mode):
        """Set the operation mode of the RBM.

        depending on the operation mode the RBM will sample the negative 
        phase of gradient descent using PCD (persistent contrative 
        divergence) or a QA (quantum annealer)

        Parameters
        ----------
        mode : {int} of value 0=op_mode_classic or 1=op_mode_quantum

        Returns
        -------
        None
        """
        if mode == self.op_mode_classic:
            self.op_mode = self.op_mode_classic
        elif mode == self.op_mode_quantum:
            self.op_mode = self.op_mode_quantum
        elif mode == self.op_mode_simulate_quantum:
            self.op_mode = self.op_mode_simulate_quantum
        else:
            raise ValueError(
                'modes can only be 0(=classic)=op_mode_classic and 1(=quantum)=op_mode_quantum.')

    def transform(self, X):
        """Compute the hidden layer activation probabilities, P(h=1|v=X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Latent representations of the data.
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        return self._mean_hiddens(X)

    def predict(self, X):
        """Compute the visible layer activation probabilities, P(v=1|v=X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        v : ndarray of shape (n_samples, n_components)
            Latent representations of the data.
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)

        _h = self._sample_hiddens(X, self.random_state_).astype(int)

        _v = self._mean_visibles(_h)

        estimation = np.zeros_like(_v)
        estimation[np.arange(len(_v)), _v.argmax(1)] = 1
        return estimation

    def _mean_hiddens(self, v):
        """Computes the probabilities P(h=1|v).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        p = safe_sparse_dot(v, self.components_.T)
        zer = np.zeros_like(p)
        p += self.intercept_hidden_
        zer += self.intercept_hidden_
        ret = expit(p, out=p)
        return ret

    def _sample_hiddens(self, v, rng):
        """Sample from the distribution P(h|v).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to sample from.

        rng : RandomState
            Random number generator to use.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer.
        """
        p = self._mean_hiddens(v)
        return (rng.random_sample(size=p.shape) < p)

    def _mean_visibles(self, h):
        """Computes the probabilities P(v=1|h).

        Parameters
        ----------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        rng : RandomState
            Random number generator to use.

        Returns
        -------
        v : ndarray of shape (n_samples, n_features)
            Corresponding mean field values for the visible layer.
        """
        p = np.dot(h, self.components_)
        p += self.intercept_visible_
        expit(p, out=p)
        return p

    def _sample_visibles(self, h, rng):
        """Sample from the distribution P(v|h).

        Parameters
        ----------
        h : ndarray of shape (n_samples, n_features)
            Values of the hidden layer to sample from.

        rng : RandomState
            Random number generator to use.

        Returns
        -------
        v : ndarray of shape (n_samples, n_components)
            Values of the visible layer.
        """
        p = self._mean_visibles(h)
        return (rng.random_sample(size=p.shape) < p)

    def _free_energy(self, v):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : ndarray of shape (n_samples,)
            The value of the free energy.
        """
        return (- safe_sparse_dot(v, self.intercept_visible_)
                - np.logaddexp(0, safe_sparse_dot(v, self.components_.T)
                               + self.intercept_hidden_).sum(axis=1))

    def gibbs(self, v):
        """Perform one Gibbs sampling step.

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to start from.

        Returns
        -------
        v_new : ndarray of shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
        check_is_fitted(self)
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        h_ = self._sample_hiddens(v, self.random_state_)
        v_ = self._sample_visibles(h_, self.random_state_)

        return v_

    def partial_fit(self, X, y=None):
        """Fit the model to the data X which should contain a partial
        segment of the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        if not hasattr(self, 'random_state_'):
            self.random_state_ = check_random_state(self.random_state)
        if not hasattr(self, 'components_'):
            self. components_ = np.asarray(
                self.random_state_.normal(
                    0,
                    0.01,
                    (self.n_components, X.shape[1])
                ),
                order='F')
        if not hasattr(self, 'intercept_hidden_'):
            self.intercept_hidden_ = np.zeros(self.n_components, )
        if not hasattr(self, 'intercept_visible_'):
            self.intercept_visible_ = np.zeros(X.shape[1], )
        if not hasattr(self, 'h_samples_'):
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        self._fit(X, self.random_state_)

    def batch_response(self, batchsize, response):
        i = 0
        ret = []
        ret_arr = []
        for datum in response.data(['sample', 'num_occurrences']):
            ret.append(datum)
            i += 1

            if i == batchsize:
                yield ret
                i = 0
                ret = []

    def split_visible_hidden(self, batch):
        num_visible = self.intercept_visible_.shape[0]

        visible_batch = []
        hidden_batch = []

        for datum in batch:
            visible = []
            hidden = []
            for i, key in enumerate(sorted(datum.sample.keys())):
                if i < num_visible:
                    visible.append(datum.sample[key])
                else:
                    hidden.append(datum.sample[key])

            visible_batch.append((tuple(visible), datum.num_occurrences))
            hidden_batch.append((tuple(hidden), datum.num_occurrences))

        return visible_batch, hidden_batch

    def most_probable(self, visible_batch):
        visible_batch_dict = {}
        for item in visible_batch:
            visible_batch_dict[item[0]] = visible_batch_dict.get(
                item[0], 0) + item[1]
        return max(visible_batch_dict, key=visible_batch_dict.get)

    def mean_batch_values(self, hidden_batch):
        total = 0
        sum_of_vectors = np.zeros(len(hidden_batch[0][0]),)
        for item in hidden_batch:
            total += item[1]
            sum_of_vectors += np.array(item[0])*item[1]
        return sum_of_vectors / total

    def _quantum_sample(self, n_samples):
        """Sample the negative phase i.e the model distribution of the RBM via a QA

        Parameters:
        -----------
        None

        Returns:
        --------
        v_neg : ndarray of shape (n_samples, n_features)
            SAMPLES of the visible layer from the model distribution.

        h_neg : ndarray of shape (n_samples, n_components)
            MEAN FIELD VALUES of the hidden layer from the model distribution.
        """
        v_neg = []
        h_neg = []
        if not hasattr(self, "dclient"):
            self.dclient = DClient()
        if self.op_mode == self.op_mode_simulate_quantum:
            self.dclient.mode = 'simulate'

        elif self.op_mode == self.op_mode_quantum:
            self.dclient.mode = 'quantum'

        else:
            raise ValueError()

        _Q = upper_diagonal_blockmatrix(
            self.intercept_visible_, self.intercept_hidden_, self.components_.T)
        Q = matrix_to_dict(_Q)

        reads_per_sample = int(10000 / n_samples)

        response = self.dclient.sample(
            Q, num_reads=reads_per_sample * n_samples)

        for batch in self.batch_response(reads_per_sample, response):
            visible_batch, hidden_batch = self.split_visible_hidden(batch)
            v_neg.append(self.mean_batch_values(visible_batch))
            h_neg.append(self.mean_batch_values(hidden_batch))

        return np.array(v_neg), np.array(h_neg)

    def _fit(self, v_pos, rng):
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        v_pos : ndarray of shape (n_samples, n_features)
            The data to use for training.

        rng : RandomState
            Random number generator to use for sampling.
        """
        h_pos = self._mean_hiddens(v_pos)

        v_neg, h_neg = None, None
        if(self.op_mode < self.op_mode_quantum):
            v_neg = self._sample_visibles(self.h_samples_, rng)
            h_neg = self._mean_hiddens(v_neg)
        else:
            v_neg, h_neg = self._quantum_sample(v_pos.shape[0])

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        self.intercept_visible_ += lr * (np.asarray(
                                         v_pos.sum(axis=0)).squeeze() -
                                         v_neg.sum(axis=0))

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)

    def score_samples(self, X):
        """Compute the pseudo-likelihood of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : ndarray of shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        check_is_fitted(self)

        v = check_array(X, accept_sparse='csr')
        rng = check_random_state(self.random_state)

        # Randomly corrupt one feature in each sample in v.
        ind = (np.arange(v.shape[0]),
               rng.randint(0, v.shape[1], v.shape[0]))
        if sp.issparse(v):
            data = -2 * v[ind] + 1
            v_ = v + sp.csr_matrix((data.A.ravel(), ind), shape=v.shape)
        else:
            v_ = v.copy()
            v_[ind] = 1 - v_[ind]

        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)
        return v.shape[1] * log_logistic(fe_ - fe)

    def fit(self, X, y=None):
        """Fit the model to the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = check_array(X, accept_sparse='csr', dtype=np.float64)
        n_samples = X.shape[0]
        rng = check_random_state(self.random_state)

        self.components_ = np.asarray(
            rng.normal(0, 0.01, (self.n_components, X.shape[1])),
            order='F')
        self.intercept_hidden_ = np.zeros(self.n_components, )
        self.intercept_visible_ = np.zeros(X.shape[1], )
        self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,
                                            n_batches, n_samples))
        verbose = self.verbose
        begin = time.time()
        self.tester.save_weights_img(
            self.intercept_visible_, self.intercept_hidden_, self.components_.T, 0)
        for iteration in range(1, self.n_iter + 1):
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], rng)

            if verbose:
                end = time.time()
                print("[%s] Iteration %d, pseudo-likelihood = %.2f,"
                      " time = %.2fs"
                      % (type(self).__name__, iteration,
                         self.score_samples(X).mean(), end - begin))
                begin = end

            self.tester.save_weights_img(
                self.intercept_visible_, self.intercept_hidden_, self.components_.T, iteration)
        return self
