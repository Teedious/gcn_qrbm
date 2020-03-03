from dwave.cloud import Client
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from dimod import StructureComposite
from neal import SimulatedAnnealingSampler
import numpy as np
import minorminer


class DClient:

    def __init__(self, mode='simulate'):
        self.mode = mode
        self.client = Client.from_config()
        self.embedding = None
        self.sampler = None
        self.base_sampler = DWaveSampler(solver={'qpu':True})

    def find_embedding(self,Q):
        if None == self.embedding:
            self.embedding = minorminer.find_embedding(Q, self.base_sampler.edgelist)
    
    def create_sampler(self):
        if None == self.sampler:
            _sampler = None
            if self.mode == 'quantum':
                _sampler = StructureComposite(self.base_sampler, self.base_sampler.nodelist, self.base_sampler.edgelist)
            else:
                _sampler = StructureComposite(SimulatedAnnealingSampler(), self.base_sampler.nodelist, self.base_sampler.edgelist)
            self.sampler = FixedEmbeddingComposite(_sampler, self.embedding)

    def sample(self,Q, num_reads=1):
        self.find_embedding(Q)
        self.create_sampler()
        answer = self.sampler.sample_qubo(Q, num_reads=num_reads)

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


    _BW = np.concatenate((B, weight_matrix),axis=1)
    _0C = np.concatenate((np.zeros_like(weight_matrix).T, C),axis=1)

    Q = np.concatenate((_BW, _0C),axis=0)
    return -1 * Q / scale_factor

def matrix_to_dict(matrix):
    matrix_dict = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix_dict[(i,j)] = matrix[i][j]

    return matrix_dict

