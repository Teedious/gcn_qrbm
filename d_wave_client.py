from dwave.cloud import Client
from dwave.cloud.exceptions import SolverFailureError
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import FixedEmbeddingComposite
from dimod import StructureComposite
from dwave_qbsolv import QBSolv
import dimod
from neal import SimulatedAnnealingSampler
from tabu import TabuSampler
import numpy as np
import minorminer

class DClient:

    op_mode_quantum = 1
    op_mode_simulated_annealing = 2
    op_mode_qbsolv = 3

    def __init__(self, mode=2):

        self.api_tokens = [
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
        i = 0
        max = 10
        while None == self.embedding and i < max:
            i += 1
            try:
                self.embedding = minorminer.find_embedding(
                    Q, self.base_sampler.edgelist)
            except Exception as a:
                if i == max-1:
                    raise Exception(
                        "No embedding found after {} tries.".format(max))

    def create_sampler(self):
        if None == self.sampler:
            composite = None
            _inner_sampler = None
            if self.mode == self.op_mode_quantum:
                _inner_sampler = self.base_sampler
            elif self.mode == self.op_mode_simulated_annealing:
                _inner_sampler = SimulatedAnnealingSampler()
            elif self.mode == self.op_mode_qbsolv:
                _inner_sampler = QBSolv()
            else:
                raise ValueError(
                    "op_mode {} is not known. (only 1,2,3)".format(self.mode))

            composite = StructureComposite(
                _inner_sampler,
                self.base_sampler.nodelist,
                self.base_sampler.edgelist)

            self.sampler = FixedEmbeddingComposite(composite, self.embedding)

    def sample(self, Q, num_reads=10):
        answer = None
        self.find_embedding(Q)
        while None == answer:
            try:
                self.create_sampler()
                # hier muss was gefixt werden
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
                # bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
                # __, target_edgelist, target_adjacency = self.sampler.target_structure
                # bqm_embedded = embed_bqm(bqm, self.embedding, target_adjacency,
                #                  smear_vartype=dimod.SPIN)
                # # embed bqm gives a binary View object of the BQM which is not allowed in the TabuSampler
                # print(isinstance(bqm, dimod.BinaryQuadraticModel))
                # print(isinstance(bqm_embedded, dimod.BinaryQuadraticModel))
                if self.mode == self.op_mode_qbsolv:
                    answer = self.sampler.sample_qubo(Q, num_repeats=num_reads-1)
                else:
                    answer = self.sampler.sample_qubo(Q, num_reads=num_reads)
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
                #######################################################################################
            except SolverFailureError:
                self.sampler = None
                print("\n\n\n{} is now empty\n\n\n".format(self.api_tokens[0]))
                with open("./empty_api_tokens", mode='a', encoding="utf-8") as file:
                    file.write("{}\n".format(self.api_tokens[0]))
                del self.api_tokens[0]
                self.base_sampler = DWaveSampler(
                    solver={'qpu': True}, token=self.api_tokens[0])
        ret = list(answer.data(['sample', 'num_occurrences']))
        return ret


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
    if len(visible_vector.shape) > 1:
        visible_vector = np.reshape(visible_vector, (-1,))

    if len(hidden_vector.shape) > 1:
        hidden_vector = np.reshape(hidden_vector, (-1,))

    B = np.diag(visible_vector)
    C = np.diag(hidden_vector)

    _BW = np.concatenate((B, weight_matrix), axis=1)
    _0C = np.concatenate((np.zeros_like(weight_matrix).T, C), axis=1)

    Q = np.concatenate((_BW, _0C), axis=0)
    return Q / scale_factor


def matrix_to_dict(matrix):
    matrix_dict = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix_dict[(i, j)] = matrix[i][j]

    return matrix_dict



import itertools

import numpy as np
import dimod

from six import iteritems, itervalues

from dwave.embedding.chain_breaks import majority_vote, broken_chains
from dwave.embedding.exceptions import MissingEdgeError, MissingChainError, InvalidNodeError
from dwave.embedding.utils import chain_to_quadratic


def embed_bqm(source_bqm, embedding, target_adjacency, chain_strength=1.0,
              smear_vartype=None):
    """Embed a binary quadratic model onto a target graph.
    Args:
        source_bqm (:obj:`.BinaryQuadraticModel`):
            Binary quadratic model to embed.
        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.
        target_adjacency (dict/:obj:`networkx.Graph`):
            Adjacency of the target graph as a dict of form {t: Nt, ...},
            where t is a variable in the target graph and Nt is its set of neighbours.
        chain_strength (float, optional):
            Magnitude of the quadratic bias (in SPIN-space) applied between
            variables to create chains, with the energy penalty of chain breaks
            set to 2 * `chain_strength`.
        smear_vartype (:class:`.Vartype`, optional, default=None):
            Determines whether the linear bias of embedded variables is smeared
            (the specified value is evenly divided as biases of a chain in the
            target graph) in SPIN or BINARY space. Defaults to the
            :class:`.Vartype` of `source_bqm`.
    Returns:
        :obj:`.BinaryQuadraticModel`: Target binary quadratic model.
    Examples:
        This example embeds a triangular binary quadratic model representing
        a :math:`K_3` clique into a square target graph by mapping variable `c`
        in the source to nodes `2` and `3` in the target.
        >>> import networkx as nx
        ...
        >>> target = nx.cycle_graph(4)
        >>> # Binary quadratic model for a triangular source graph
        >>> h = {'a': 0, 'b': 0, 'c': 0}
        >>> J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}
        >>> bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
        >>> # Variable c is a chain
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> # Embed and show the chain strength
        >>> target_bqm = dwave.embedding.embed_bqm(bqm, embedding, target)
        >>> target_bqm.quadratic[(2, 3)]
        -1.0
        >>> print(target_bqm.quadratic)  # doctest: +SKIP
        {(0, 1): 1.0, (0, 3): 1.0, (1, 2): 1.0, (2, 3): -1.0}
    See also:
        :func:`.embed_ising`, :func:`.embed_qubo`
    """
    if smear_vartype is dimod.SPIN and source_bqm.vartype is dimod.BINARY:
        return embed_bqm(source_bqm.spin, embedding, target_adjacency,
                         chain_strength=chain_strength, smear_vartype=None).binary
    elif smear_vartype is dimod.BINARY and source_bqm.vartype is dimod.SPIN:
        return embed_bqm(source_bqm.binary, embedding, target_adjacency,
                         chain_strength=chain_strength, smear_vartype=None).spin

    # create a new empty binary quadratic model with the same class as source_bqm
    try:
        target_bqm = source_bqm.base.empty(source_bqm.vartype)
    except AttributeError:
        # dimod < 0.9.0
        target_bqm = source_bqm.empty(source_bqm.vartype)

    # add the offset
    target_bqm.add_offset(source_bqm.offset)

    # start with the linear biases, spreading the source bias equally over the target variables in
    # the chain
    for v, bias in iteritems(source_bqm.linear):

        if v in embedding:
            chain = embedding[v]
        else:
            raise MissingChainError(v)

        if any(u not in target_adjacency for u in chain):
            raise InvalidNodeError(v, next(u not in target_adjacency for u in chain))

        try:
            b = bias / len(chain)
        except ZeroDivisionError:
            raise MissingChainError(v)

        target_bqm.add_variables_from({u: b for u in chain})

    # next up the quadratic biases, spread the quadratic biases evenly over the available
    # interactions
    for (u, v), bias in iteritems(source_bqm.quadratic):
        available_interactions = {(s, t) for s in embedding[u] for t in embedding[v] if s in target_adjacency[t]}

        if not available_interactions:
            raise MissingEdgeError(u, v)

        b = bias / len(available_interactions)

        target_bqm.add_interactions_from((u, v, b) for u, v in available_interactions)

    for chain in itervalues(embedding):

        # in the case where the chain has length 1, there are no chain quadratic biases, but we
        # none-the-less want the chain variables to appear in the target_bqm
        if len(chain) == 1:
            v, = chain
            target_bqm.add_variable(v, 0.0)
            continue

        quadratic_chain_biases = chain_to_quadratic(chain, target_adjacency, chain_strength)
        # this is in spin, but we need to respect the vartype
        if target_bqm.vartype is dimod.SPIN:
            target_bqm.add_interactions_from(quadratic_chain_biases)
        else:
            # do the vartype converstion
            for (u, v), bias in quadratic_chain_biases.items():
                target_bqm.add_interaction(u, v, 4*bias)
                target_bqm.add_variable(u, -2*bias)
                target_bqm.add_variable(v, -2*bias)
                target_bqm.add_offset(bias)

        # add the energy for satisfied chains to the offset
        energy_diff = -sum(itervalues(quadratic_chain_biases))
        target_bqm.add_offset(energy_diff)

    return target_bqm