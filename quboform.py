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
import networkx as nx
import matplotlib.pyplot as plt
from d_wave_client import upper_diagonal_blockmatrix

def save_weights_img( B= None, C= None, W= None):
        #Q = upper_diagonal_blockmatrix(
        #    B, C, W)
        heatmap = plt.imshow(W, cmap='PiYG', interpolation='nearest', clim=(
            -1, 1),)
        plt.colorbar(heatmap)
        plt.show()
        plt.clf()

base_sampler = DWaveSampler(solver={'qpu': True}, token="DEV-6a4c8dd1862e3e593ab8ee3e1389184107f52c45")

G = nx.Graph()
G.add_edges_from(base_sampler.edgelist)
W = nx.to_numpy_array(G)

a = set(range(2048))
b = set(base_sampler.nodelist)
c = np.array(sorted(a.intersection(b)))
d = sorted(a-b)

B_ = [a % 8 < 4 for a in c]
B = c[B_]
C_ = [a % 8 >=4 for a in c]
C = c[C_]

save_weights_img(W=W)
