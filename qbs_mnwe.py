
from dwave.system.composites import FixedEmbeddingComposite
from dimod import StructureComposite
from neal import SimulatedAnnealingSampler
from tabu import TabuSampler
from dwave_qbsolv import QBSolv
import numpy as np
import minorminer

# testin function
def show_if_works(sampler):
    qubo = {(0,0): -1, (0,1):2, (1,1):-1}
    embedding = {0:[0], 1:[1]}
    composite = StructureComposite(sampler,[0,1],[(0,1)])
    fixed = FixedEmbeddingComposite(composite, embedding)

    print("\n\n")
    #test standard -> should deliver 100 samples
    print(fixed.sample_qubo(qubo, num_reads = 100))

    #test special case qbsolv -> should deliver 100 samples also for qbsolv
    print(fixed.sample_qubo(qubo, num_reads = 100, num_repeats = 99))


# this should just work
show_if_works(SimulatedAnnealingSampler())

# here the number of samples is wrong
#
# The problem is the naming of the parameter
# num_reads vs. num_repeats
#
# num_repeats in documentation:
# num_repeats (int, optional) – Determines the number of times to
# repeat the main loop in qbsolv after determining a better sample. 
# Default 50. 
show_if_works(QBSolv())


# this will throw an error
show_if_works(TabuSampler())