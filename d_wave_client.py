from dwave.cloud import Client
from dwave.cloud.exceptions import SolverFailureError
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from dimod import StructureComposite
from neal import SimulatedAnnealingSampler
import numpy as np
import minorminer


class DClient:

    def __init__(self, mode='simulate'):
        self.api_tokens = [
            "DEV-67f1762d26c10ae03bc0e6b61d6216ebbea8e03e",
            "DEV-6a4c8dd1862e3e593ab8ee3e1389184107f52c45",
            "DEV-2fd0bc12a079b0089fb4c8bd4e439694160945ce",
            "DEV-7f56a1d2d5504b12b6bc7df346e2b18d59bea4a3",
            "DEV-e2bc6c527b3b6402000edcd3fa6d9c572a1ee872",
            "DEV-0cbee57b56cbe00d520a1dcd4d57fbde62f56869",
            "DEV-650859a4adf3a4fdee3e958ca4ab4ed45c17e393",
            "DEV-650859a4adf3a4fdee3e958ca4ab4ed45c17e393",
        ]
        self.mode = mode
        self.embedding = None
        self.sampler = None
        self.base_sampler = DWaveSampler(
            solver={'qpu': True}, token=self.api_tokens[0])

    def find_embedding(self, Q):
        if None == self.embedding:
            self.embedding = minorminer.find_embedding(
                Q, self.base_sampler.edgelist)

    def create_sampler(self):
        if None == self.sampler:
            _sampler = None
            if self.mode == 'quantum':
                _sampler = StructureComposite(
                    self.base_sampler, self.base_sampler.nodelist, self.base_sampler.edgelist)
                _sampler.properties['default_annealing_time'] = 1
            else:
                _sampler = StructureComposite(SimulatedAnnealingSampler(
                ), self.base_sampler.nodelist, self.base_sampler.edgelist)
            self.sampler = FixedEmbeddingComposite(_sampler, self.embedding)

    def sample(self, Q, num_reads=10):
        answer = None
        self.find_embedding(Q)
        while None == answer:
            try:
                self.create_sampler()
                answer = self.sampler.sample_qubo(Q, num_reads=num_reads)
            except SolverFailureError:
                self.sampler = None
                del self.api_tokens[0]
                self.base_sampler = DWaveSampler(
                    solver={'qpu': True}, token=self.api_tokens[0])

        return answer


def upper_diagonal_blockmatrix(visible_vector, hidden_vector, weight_matrix, scale_factor=1.):
    """Generate an upder triangle matrix with visible and hidden vectors
    as diagonal and weight matrix as upper block matrix.

    [v11,v22], [h11,h22,h33],            

    [[w11,w12,w13],

    [w21,w22,w23]]              

    -->

    [

    [v11,____,w11,w12,w13],

    [____,v22,w21,w22,w23],

    [____,____,_h11,____,____],

    [____,____,____, h22,____],

    [____,____,____,____,_h33]

    ]

    """

    B = np.diag(visible_vector)
    C = np.diag(hidden_vector)

    _BW = np.concatenate((B, weight_matrix), axis=1)
    _0C = np.concatenate((np.zeros_like(weight_matrix).T, C), axis=1)

    Q = np.concatenate((_BW, _0C), axis=0)
    return -1 * Q / scale_factor


def matrix_to_dict(matrix):
    matrix_dict = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix_dict[(i, j)] = matrix[i][j]

    return matrix_dict
