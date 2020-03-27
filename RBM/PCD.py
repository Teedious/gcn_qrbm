#####################################################
#
# A restricted Boltzmann machine trained using the
# constrastive divergence algorithm
#
# see:
#
# G. Hinton, Training products of experts by
# minimizing contrastive divergence, Journal Neural
# Computation Vol. 14, No. 8 (2002), 1771--1800
#
# G.~Hinton,
# A practical guide to training restricted Boltzmann
# machines, Technical Report University of Montreal
# TR-2010-003 (2010)
#
#
# Copyright (c) 2018 christianb93
# Permission is hereby granted, free of charge, to
# any person obtaining a copy of this software and
# associated documentation files (the "Software"),
# to deal in the Software without restriction,
# including without limitation the rights to use,
# copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice
# shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY
# OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#####################################################

import numpy as np
from scipy.special import expit
from . import Base
from d_wave_client import *
import matplotlib.pyplot as plt


class PCDRBM (Base.BaseRBM):

    op_mode_classic = 0
    op_mode_quantum = 1
    op_mode_simulate_quantum = 2

    def __init__(self, visible=8, hidden=3, particles=10, beta=2.0, precision=64, iterations=100, epochs=1, step=0.001, weight_decay=0.0001, op_mode=0):
        self.visible = visible
        self.hidden = hidden
        self.beta = beta
        self.particles = particles
        self.iterations = iterations
        self.epochs = epochs
        self.step = step
        self.weight_decay = weight_decay
        self.op_mode = op_mode

        if precision == 64:
            self.np_type = np.float64
        elif precision == 32:
            self.np_type = np.float32
        else:
            raise ValueError("Unsupported precisions")
        self.precision = precision
        #
        # Initialize weights with a random normal distribution
        #
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(visible, hidden))
        #
        # set bias to zero
        #
        self.b = np.zeros(dtype=float, shape=(1, visible))
        self.c = np.zeros(dtype=float, shape=(1, hidden))
        #
        # Initialize the particles
        #
        if -1 != particles:
            self.N = np.random.randint(low=0, high=2, size=(
                particles, self.visible)).astype(int)
        self.global_step = 0

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

    #
    # Train the model on a training data mini batch
    # stored in V. Each row of V corresponds to one
    # sample. The number of columns of V should
    # be equal to the number of visible units
    #
    def train(self, V, iterations=100, epochs=1, step=0.001, weight_decay=0.0001):
        #
        # Check geometry
        #
        batch_size = V.shape[0]
        if -1 == self.particles:
            self.particles = batch_size
            self.N = np.random.randint(low=0, high=2, size=(
                self.particles, self.visible)).astype(int)

        if (V.shape[1] != self.visible):
            print("Shape of training data", V.shape)
            raise ValueError("Data does not match number of visible units")
        initial_step_size = step
        #
        # Prepare logs
        #
        dw = []
        errors = []
        # self.save_weights_img(self.global_step)
        for i in range(iterations):
            #
            # Update step size - we do this linearly over time
            #
            step = initial_step_size * \
                (1.0 - (1.0*self.global_step)/(1.0*iterations*epochs))
            #
            # First we compute the negative phase. We run the
            # Gibbs sampler for one step, starting at the previous state
            # of the particles self.N
            #
            self.N, _ = self.runGibbsStep(self.N, size=self.particles)

            Eb, neg = None, None
            if self.op_mode == self.op_mode_classic:
                #
                # and use this to calculate the negative phase
                #
                Eb = expit(self.beta*(np.matmul(self.N, self.W) +
                                      self.c), dtype=self.np_type)
                neg = np.tensordot(self.N, Eb, axes=(
                    (0), (0))).astype(self.np_type)
            else:
                V_neg, Eb = self._quantum_sample(V.shape[0])
                neg = np.tensordot(V_neg, Eb, axes=(
                    (0), (0))).astype(self.np_type)

            #
            # Now we compute the positive phase. We need the
            # expectation values of the hidden units
            #
            E = expit(self.beta*(np.matmul(V, self.W) + self.c))
            pos = np.tensordot(V, E, axes=((0), (0)))
            #
            # Now update weights
            #
            dW = step*self.beta*(pos - neg) / float(batch_size) - \
                step*weight_decay*self.W / float(batch_size)
            self.W += dW
            self.b += step*self.beta*np.sum(V - self.N, 0) / float(batch_size)
            self.c += step*self.beta*np.sum(E - Eb, 0) / float(batch_size)
            dw.append(np.linalg.norm(dW))
            #
            # Compute reconstruction error every few iterations
            #
            if 0 == (self.global_step % 50):
                Vb = self.sampleFrom(initial=V, size=batch_size, iterations=1)
                recon_error = np.linalg.norm(V - Vb)
                errors.append(recon_error)
                # if 0 == (self.global_step % 500):
                # print("Iteration ",self.global_step,"recon error is ", recon_error)
                print("Iteration ", self.global_step,
                      "recon error is ", recon_error)

            self.global_step += 1
            # self.skave_weights_img(self.global_step)
        return dw, errors

    def save_weights_img(self, i):
        __Q = upper_diagonal_blockmatrix(
            self.b, self.c, self.W)
        heatmap = plt.imshow(__Q, cmap='hot', interpolation='nearest')
        plt.colorbar(heatmap)
        plt.savefig(
            'metrics/plt/new_rbm_{}_{}.png'.format(self.op_mode, i))
        plt.clf()

    def recover(self, V, iterations):
        batch_size = V.shape[0]
        for _ in range(iterations):
            V, H = self.runGibbsStep(V, size=batch_size)
        return V, H

    def transform(self, V):
        _, H = self.recover(V, 100)
        return H

    def fit(self, x_train, y_train):
        self.train(x_train, self.iterations, self.epochs,
                   self.step, self.weight_decay)
        return self

    def predict(self, x_test):
        V, _ = self.recover(x_test, 100)
        return V

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
        num_visible = self.b.shape[1]

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
            self.b, self.c, self.W)
        Q = matrix_to_dict(_Q)

        reads_per_sample = int(10000 / n_samples)

        response = self.dclient.sample(
            Q, num_reads=reads_per_sample * n_samples)

        for batch in self.batch_response(reads_per_sample, response):
            visible_batch, hidden_batch = self.split_visible_hidden(batch)
            v_neg.append(self.mean_batch_values(visible_batch))
            h_neg.append(self.mean_batch_values(hidden_batch))

        return np.array(v_neg), np.array(h_neg)
